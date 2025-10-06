import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- NEW: Imports for Audio and Hugging Face ---
import librosa
from transformers import AutoProcessor, AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss

# ---------------- Config / Seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- MODIFIED: Use CLAP backbone ---
BACKBONE_NAME = "laion/clap-htsat-unfused"

# --- MODIFIED: Update Data Paths for audio ---
TRAIN_DIR = "../final_data1/train/audio"
VAL_DIR = "../final_data1/val/audio"

BATCH_SIZE = 32 # You may need to lower this if you run out of GPU memory
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 10
MIN_DELTA = 1e-4
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1

# --- NEW: Audio specific config ---
SAMPLE_RATE = 48000 # Required sample rate for the CLAP model
AUDIO_DURATION_S = 10 # Duration to pad/truncate audio to (in seconds)
N_SAMPLES = AUDIO_DURATION_S * SAMPLE_RATE

CHECKPOINT_PATH = "best_clap_linear_head_checkpoint.pt"

# ---------------- Load CLAP model and processor ----------------
print(f"Loading CLAP processor and model: {BACKBONE_NAME}...")
processor = AutoProcessor.from_pretrained(BACKBONE_NAME)
clap_model = AutoModel.from_pretrained(BACKBONE_NAME, use_safetensors=True).to(DEVICE)
clap_model.eval()

# --- NEW: Custom Audio Dataset ---
class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, target_sr, target_samples):
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.target_samples = target_samples
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for audio_path in class_dir.iterdir():
                if audio_path.suffix.lower() in audio_extensions:
                    self.samples.append((str(audio_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load audio and resample if necessary
        waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Pad or truncate to target length
        if len(waveform) > self.target_samples:
            waveform = waveform[:self.target_samples]
        else:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)), 'constant')
            
        return waveform, label

# --- NEW: Collate function to apply processor to a batch ---
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    inputs = processor(audios=list(waveforms), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs['labels'] = torch.tensor(labels)
    return inputs

# ---------------- Datasets & Sampler ----------------
train_ds = AudioFolderDataset(TRAIN_DIR, target_sr=SAMPLE_RATE, target_samples=N_SAMPLES)
val_ds   = AudioFolderDataset(VAL_DIR, target_sr=SAMPLE_RATE, target_samples=N_SAMPLES)

labels = [label for _, label in train_ds.samples]
class_counts = Counter(labels)
num_samples = len(train_ds)
class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

NUM_CLASSES = len(train_ds.classes)

# --- MODIFIED: CLAP Model with Classification Head ---
class CLAPClassificationHead(nn.Module):
    def __init__(self, clap_model, n_cls, dropout_rate=0.5):
        super().__init__()
        self.clap = clap_model
        
        for param in self.clap.parameters():
            param.requires_grad = False
            
        proj_dim = self.clap.config.audio_config.hidden_size # Get output dim from config
        hidden_dim = proj_dim // 2
        
        self.head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_cls)
        )

    def forward(self, input_features):
        audio_features = self.clap.get_audio_features(input_features)
        return self.head(audio_features)

# ---------------- Instantiate Model ----------------
model = CLAPClassificationHead(clap_model, NUM_CLASSES, dropout_rate=DROPOUT).to(DEVICE)

# Ensure only the head is trainable
for name, param in model.named_parameters():
    if 'head' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/total:.2f}%)")

# ---------------- Optimizer / Loss / AMP / Scheduler ----------------
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- MODIFIED: Evaluation Functions for CLAP ---
@torch.no_grad()
def zero_shot_acc():
    clap_model.eval()
    templates = [f"a sound of a {c}" for c in train_ds.classes]
    text_inputs = processor(text=templates, return_tensors="pt", padding=True)
    text_features = clap_model.get_text_features(
        input_ids=text_inputs['input_ids'].to(DEVICE),
        attention_mask=text_inputs['attention_mask'].to(DEVICE)
    )
    
    correct = total = 0
    for batch in val_loader:
        audio_features = clap_model.get_audio_features(batch['input_features'].to(DEVICE))
        logits = (audio_features @ text_features.T)
        preds = logits.argmax(dim=-1)
        correct += (preds == batch['labels'].to(DEVICE)).sum().item()
        total += batch['labels'].size(0)
    return 100.0 * correct / total

@torch.no_grad()
def evaluate(net, print_report=False):
    net.eval()
    loss_sum = 0.0
    all_preds, all_labels, probs_list = [], [], []
    for batch in val_loader:
        input_features = batch['input_features'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        with torch.amp.autocast(device_type=DEVICE):
            out = net(input_features)
            loss = criterion(out, labels)
            
        loss_sum += loss.item() * labels.size(0)
        probs = torch.nn.functional.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        probs_list.append(probs.cpu().numpy())

    total = len(all_labels)
    avg_loss = loss_sum / total
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / total * 100.0
    probs_all = np.concatenate(probs_list, axis=0)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    logloss = log_loss(all_labels, probs_all, labels=list(range(NUM_CLASSES)))

    if print_report:
        print(f"Val loss: {avg_loss:.4f} | Acc: {acc:.2f}% | F1: {f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=val_ds.classes, zero_division=0))

    return avg_loss, acc, prec, rec, f1, logloss

# ---------------- Training Loop ----------------
mlflow.set_experiment("clap_htsat_linear_probe")
with mlflow.start_run():
    mlflow.log_params({
        "data_split": "audio_files",
        "backbone": BACKBONE_NAME, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "dropout": DROPOUT, "label_smoothing": LABEL_SMOOTHING,
        "architecture": "Frozen CLAP + MLP Head"
    })

    zs_acc = zero_shot_acc()
    print(f"Zero-shot accuracy: {zs_acc:.2f}%")
    mlflow.log_metric("zero_shot_acc", zs_acc)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            # --- MODIFIED: Get data from batch dictionary ---
            input_features = batch['input_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE):
                out = model(input_features)
                loss = criterion(out, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            running_correct += (out.argmax(dim=1) == labels).sum().item()
            running_samples += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_correct/running_samples:.2f}%")

        train_loss = running_loss / running_samples
        train_acc  = 100.0 * running_correct / running_samples
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_logloss = evaluate(model)
        scheduler.step()

        mlflow.log_metrics({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Saved improved checkpoint to {CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation.")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        
        final_metrics = evaluate(model, print_report=True)
        mlflow.log_metrics({
            "final_val_loss": final_metrics[0], "final_val_acc": final_metrics[1],
            "final_val_precision": final_metrics[2], "final_val_recall": final_metrics[3],
            "final_val_f1": final_metrics[4], "final_val_logloss": final_metrics[5]
        })
        mlflow.pytorch.log_model(model, "clap_linear_head_model")

print("\nDone.")