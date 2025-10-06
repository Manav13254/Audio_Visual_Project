import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ---------------- Config / Seed ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "../final_data/train/spectrograms"
VAL_DIR = "../final_data/val/spectrograms"

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 100
# --- CHANGE 1: Higher learning rate is needed for training from scratch ---
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 20 # Increased patience as scratch training takes much longer
MIN_DELTA = 5e-4

CHECKPOINT_PATH = "best_rn18_from_scratch_checkpoint.pt"

# ---------------- Helper Function to Calculate Dataset Stats ----------------
def get_dataset_mean_std(data_dir):
    """
    Calculates the mean and standard deviation of a dataset of images.
    This is necessary for normalizing the data when not using a pre-trained model.
    """
    print("Calculating dataset mean and std...")
    # Use a simple transform to convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in tqdm(loader, desc="Calculating Stats"):
        batch_samples = images.size(0) # batch size (the last batch can be smaller)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
    return mean.tolist(), std.tolist()

# --- Main execution block ---
if __name__ == '__main__':
    # --- CHANGE 2: Calculate our own normalization stats for our dataset ---
    # We only need to do this once. For subsequent runs, you can hardcode the values.
    dataset_mean, dataset_std = get_dataset_mean_std(TRAIN_DIR)

    # ---------------- Transforms for Spectrogram Images ----------------
    # --- CHANGE 3: Use our calculated stats for normalization ---
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    
    # --- The rest of the script is similar, with changes to the model loading ---
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    labels = [label for _, label in train_ds.samples]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    NUM_CLASSES = len(train_ds.classes)
    print(f"Using device: {DEVICE}")
    print(f"Found {NUM_CLASSES} classes: {', '.join(train_ds.classes)}")

    # ---------------- Model (Full Fine-tuning of ResNet-18) ----------------
    print("\nLoading ResNet-18 model with RANDOM weights (training from scratch)...")
    # --- CHANGE 4: `weights=None` is the key change for training from scratch ---
    model = models.resnet18(weights=None, num_classes=NUM_CLASSES)
    
    # In this case, we don't need to replace the head, as we can specify `num_classes` directly.
    # However, to keep the architecture consistent with your more complex head, let's build it manually.
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
       nn.Linear(num_ftrs, num_ftrs // 2),
       nn.ReLU(),
       nn.Dropout(p=0.7),
       nn.Linear(num_ftrs // 2, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # All parameters are trainable by default when training from scratch
    for param in model.parameters():
        param.requires_grad = True

    # ---------------- Optimizer / Loss / Scheduler ----------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DEVICE == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------------- Evaluation Function ----------------
    def evaluate(loader, model, criterion, class_names, device, print_report=False):
        model.eval()
        running_loss, running_correct, running_samples = 0.0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device):
                    out = model(imgs)
                    loss = criterion(out, labels)
                running_loss += loss.item() * imgs.size(0)
                preds = out.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_samples += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = running_loss / running_samples
        avg_acc = 100.0 * running_correct / running_samples
        f1 = f1_score(all_labels, all_preds, average="macro")
        if print_report:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
        return avg_loss, avg_acc, f1

    # ---------------- MLflow and Training Loop ----------------
    mlflow.set_experiment("sound_classification_rn18_from_scratch")
    
    with mlflow.start_run():
        mlflow.log_params({
            "training_type": "From Scratch",
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, 
            "architecture": "ResNet-18",
            "augmentation": "TrivialAugmentWide + RandomErasing",
            "regularization": f"Dropout (p=0.7), WeightDecay ({WEIGHT_DECAY}), LabelSmoothing (0.1)"
        })
        
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            # ... (The training loop is identical) ...
            running_loss, running_correct, running_samples = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
            
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=DEVICE):
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * imgs.size(0)
                running_correct += (out.argmax(dim=1) == labels).sum().item()
                running_samples += imgs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*running_correct/running_samples:.2f}%")

            train_loss = running_loss / running_samples
            train_acc = 100.0 * running_correct / running_samples
            current_lr = optimizer.param_groups[0]['lr']
            
            val_loss, val_acc, val_f1 = evaluate(val_loader, model, criterion, val_ds.classes, DEVICE)
            scheduler.step()

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc, 
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "learning_rate": current_lr
            }, step=epoch)

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> Saved improved checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("\nLoading best model for final evaluation...")
        if Path(CHECKPOINT_PATH).exists():
            model.load_state_dict(torch.load(CHECKPOINT_PATH))
            final_loss, final_acc, final_f1 = evaluate(val_loader, model, criterion, val_ds.classes, DEVICE, print_report=True)
            mlflow.log_metrics({"final_val_loss": final_loss, "final_val_acc": final_acc, "final_val_f1": final_f1})
            mlflow.pytorch.log_model(model, "sound_classifier_rn18_from_scratch_model")
        else:
            print("No checkpoint was saved.")

    print("\nDone.")