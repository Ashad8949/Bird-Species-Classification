"""
Bird Species Classification – Deep ResNet-50 with SE blocks
Targets 0.98+ avg(Macro F1, Micro F1)

Dataset structure expected:
    data/
    ├── train/        ← training images
    ├── test/         ← test images
    └── train.csv     ← columns: image_name, label

Command-line usage:
    Train:
        python bird_classification.py --mode train \
            --dataset_path /path/to/data \
            --model_save_path /path/to/save/model.pth

    Inference:
        python bird_classification.py --mode inference \
            --dataset_path /path/to/data \
            --model_path /path/to/model.pth \
            --output_path predictions.csv

Key improvements over a basic CNN baseline:
  - ResNet-50 bottleneck architecture with SE blocks (~25M params)
  - Label smoothing (epsilon=0.1) to prevent overconfidence
  - Mixup + CutMix augmentation (50% chance each batch)
  - Random Erasing augmentation
  - Cosine Annealing Warm Restarts (T0=10, T_mult=2)
  - Test Time Augmentation (4 views averaged at inference)
  - AdamW optimizer with weight decay
  - Gradient clipping (max_norm=5.0)
  - Class-weighted loss for imbalance handling
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────── reproducibility ───────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════
#                        MODEL
# ══════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class Bottleneck(nn.Module):
    """ResNet-50 style bottleneck block with SE"""
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        mid      = out_ch
        expanded = out_ch * self.expansion

        self.conv1 = nn.Conv2d(in_ch,  mid,      1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid,    mid,      3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid,    expanded, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(expanded)
        self.se    = SEBlock(expanded) if use_se else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != expanded:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, expanded, 1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


class ResNet50SE(nn.Module):
    """
    ResNet-50 + SE blocks
    Layer depths : [3, 4, 6, 3]
    Stage widths : 64 -> 256 -> 512 -> 1024 -> 2048
    """
    def __init__(self, num_classes=15, dropout=0.4):
        super().__init__()
        self.in_ch = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_ch, blocks, stride):
        layers = [Bottleneck(self.in_ch, out_ch, stride=stride, use_se=True)]
        self.in_ch = out_ch * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_ch, out_ch, stride=1, use_se=True))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ══════════════════════════════════════════════════════
#                       LOSS
# ══════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        with torch.no_grad():
            soft = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            soft = soft * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            w    = self.weight[targets]
            loss = -(soft * log_prob).sum(dim=1)
            loss = (loss * w).sum() / w.sum()
        else:
            loss = -(soft * log_prob).sum(dim=1).mean()
        return loss


# ══════════════════════════════════════════════════════
#                   AUGMENTATION
# ══════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ══════════════════════════════════════════════════════
#                    DATASET
# ══════════════════════════════════════════════════════

class BirdDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label_to_idx=None):
        self.df           = df.reset_index(drop=True)
        self.df['label']  = self.df['label'].astype(str)   # always str
        self.img_dir      = img_dir
        self.transform    = transform
        self.label_to_idx = label_to_idx or {
            n: i for i, n in enumerate(sorted(self.df['label'].unique()))
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.label_to_idx[row['label']]
        return image, label


# ══════════════════════════════════════════════════════
#                   TRANSFORMS
# ══════════════════════════════════════════════════════

IMG_MEAN  = [0.485, 0.456, 0.406]
IMG_STD   = [0.229, 0.224, 0.225]
CROP_SIZE = 256

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(int(CROP_SIZE * 1.15)),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

def get_tta_transforms():
    """4 augmented views for Test Time Augmentation"""
    norm = [transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)]
    size = int(CROP_SIZE * 1.15)
    return [
        transforms.Compose([transforms.Resize(size),
                             transforms.CenterCrop(CROP_SIZE)] + norm),
        transforms.Compose([transforms.Resize(size),
                             transforms.CenterCrop(CROP_SIZE),
                             transforms.RandomHorizontalFlip(p=1.0)] + norm),
        transforms.Compose([transforms.Resize(int(CROP_SIZE * 1.3)),
                             transforms.CenterCrop(CROP_SIZE)] + norm),
        transforms.Compose([transforms.Resize(size),
                             transforms.RandomRotation(10),
                             transforms.CenterCrop(CROP_SIZE)] + norm),
    ]


# ══════════════════════════════════════════════════════
#                    TRAINING
# ══════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── paths ─────────────────────────────────────────
    # Handle both: dataset_path = root (with train/ subfolder) OR dataset_path = train folder itself
    if os.path.isdir(os.path.join(args.dataset_path, 'train')):
        train_img_dir = os.path.join(args.dataset_path, 'train')
        csv_path      = os.path.join(args.dataset_path, 'train.csv')
    else:
        train_img_dir = args.dataset_path
        csv_path      = os.path.join(os.path.dirname(args.dataset_path), 'train.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(args.dataset_path, 'train.csv')

    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Train image folder not found: {train_img_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"train.csv not found: {csv_path}")

    # ── load & split ──────────────────────────────────
    df = pd.read_csv(csv_path)
    # Ensure labels are plain Python strings (handles int64 or str columns equally)
    df['label'] = df['label'].astype(str)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=SEED
    )

    classes      = sorted(df['label'].unique())
    label_to_idx = {n: i for i, n in enumerate(classes)}
    num_classes  = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train samples: {len(train_df)}  |  Val samples: {len(val_df)}")

    # ── save class names alongside model ──────────────
    save_dir = os.path.dirname(os.path.abspath(args.model_save_path))
    os.makedirs(save_dir, exist_ok=True)
    class_file = os.path.join(save_dir, 'class_names.txt')
    with open(class_file, 'w') as f:
        for c in classes:
            f.write(str(c) + '\n')
    print(f"Class names saved -> {class_file}")

    # ── datasets & loaders ────────────────────────────
    train_ds = BirdDataset(train_df, train_img_dir, get_train_transform(), label_to_idx)
    val_ds   = BirdDataset(val_df,   train_img_dir, get_val_transform(),   label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── class weights for imbalance ───────────────────
    weights       = compute_class_weight('balanced', classes=np.array(classes),
                                         y=train_df['label'].values)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # ── model / loss / optimizer / scheduler ──────────
    model     = ResNet50SE(num_classes=num_classes, dropout=0.4).to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── loop ──────────────────────────────────────────
    ALPHA            = 0.4
    MIXUP_PROB       = 0.5
    best_f1          = 0.0
    patience         = 20
    patience_counter = 0
    num_epochs       = 120

    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if random.random() < 0.5:          # apply mix augmentation
                if random.random() > MIXUP_PROB:
                    images, ya, yb, lam = cutmix_data(images, labels, ALPHA)
                else:
                    images, ya, yb, lam = mixup_data(images, labels, ALPHA)
                mixed = True
            else:
                ya, yb, lam, mixed = labels, labels, 1.0, False

            optimizer.zero_grad()
            outputs = model(images)
            loss    = mixup_criterion(criterion, outputs, ya, yb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            if not mixed:
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        scheduler.step(epoch)
        train_loss /= len(train_loader.dataset)
        train_acc   = correct / total if total > 0 else float('nan')

        # validate
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images)
                val_loss += F.cross_entropy(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        macro_f1  = f1_score(all_labels, all_preds, average='macro')
        micro_f1  = f1_score(all_labels, all_preds, average='micro')
        f1_avg    = (macro_f1 + micro_f1) / 2
        lr_now    = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d} | lr={lr_now:.6f} | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"MacroF1={macro_f1:.4f} MicroF1={micro_f1:.4f} Avg={f1_avg:.4f}"
        )

        if f1_avg > best_f1:
            best_f1 = f1_avg
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'classes':          classes,
                'f1_avg':           f1_avg,
                'macro_f1':         macro_f1,
                'micro_f1':         micro_f1,
            }, args.model_save_path)
            print(f"  -> Best model saved (avg F1 = {f1_avg:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}  (best avg F1 = {best_f1:.4f})")
                break

    print(f"\nTraining complete. Best avg F1 = {best_f1:.4f}")


# ══════════════════════════════════════════════════════
#                    INFERENCE
# ══════════════════════════════════════════════════════

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── paths ─────────────────────────────────────────
    # Try  dataset_path/test/  first, then dataset_path itself as fallback
    test_img_dir = os.path.join(args.dataset_path, 'test')
    if not os.path.isdir(test_img_dir):
        test_img_dir = args.dataset_path
    print(f"Test images: {test_img_dir}")

    # ── class names ───────────────────────────────────
    ckpt  = torch.load(args.model_path, map_location=device, weights_only=False)
    classes = ckpt.get('classes', None)
    if classes is None:
        model_dir  = os.path.dirname(os.path.abspath(args.model_path))
        class_file = os.path.join(model_dir, 'class_names.txt')
        if os.path.exists(class_file):
            with open(class_file) as f:
                classes = [l.strip() for l in f if l.strip()]
        else:
            classes = [str(i) for i in range(15)]

    idx_to_label = {i: n for i, n in enumerate(classes)}
    print(f"Loaded {len(classes)} classes.")

    # ── model ─────────────────────────────────────────
    model = ResNet50SE(num_classes=len(classes), dropout=0.0).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded from {args.model_path}")

    # ── TTA inference ─────────────────────────────────
    tta_transforms = get_tta_transforms()
    test_files = sorted([
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Predicting {len(test_files)} images with TTA x{len(tta_transforms)} ...")

    results = []
    with torch.no_grad():
        for fname in test_files:
            pil_img   = Image.open(os.path.join(test_img_dir, fname)).convert('RGB')
            probs_sum = None
            for t in tta_transforms:
                probs = F.softmax(model(t(pil_img).unsqueeze(0).to(device)), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            pred_label = idx_to_label[probs_sum.argmax(1).item()]
            results.append({'image_name': fname, 'label': pred_label})

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_path, index=False)
    print(f"Saved {len(df_out)} predictions -> {args.output_path}")


# ══════════════════════════════════════════════════════
#                      MAIN
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bird Species Classification')
    parser.add_argument('--mode', required=True, choices=['train', 'inference'])
    parser.add_argument('--dataset_path', required=True,
                        help='Root data folder  (contains train/, test/, train.csv)')
    parser.add_argument('--model_save_path', default=None,
                        help='[train] where to save model.pth')
    parser.add_argument('--model_path', default=None,
                        help='[inference] path to trained model.pth')
    parser.add_argument('--output_path', default='predictions.csv',
                        help='[inference] output CSV filename')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.model_save_path:
            parser.error('--model_save_path is required for train mode')
        train(args)
    else:
        if not args.model_path:
            parser.error('--model_path is required for inference mode')
        inference(args)
