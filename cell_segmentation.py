"""
Improved Multi-class Cell Segmentation using Attention UNet
===========================================================
Key improvements over v1:
- Attention UNet with residual conv blocks & SE blocks
- Deep supervision
- Elastic deformation + stain augmentation
- Focal + Dice loss combination
- Warmup + Cosine LR, gradient clipping
- Mixed precision (AMP) training
- Test-time augmentation (TTA)
"""

import os, argparse, random, math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5
COLOR_MAP = {0:(0,0,0), 1:(255,255,0), 2:(255,0,0), 3:(0,255,0), 4:(0,0,255)}
RGB_TO_CLASS = {v:k for k,v in COLOR_MAP.items()}
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ─── Mask Conversion ─────────────────────────────────────────────────────────

def rgb_mask_to_class(mask_np):
    h, w, _ = mask_np.shape
    class_map = np.zeros((h, w), dtype=np.int64)
    for rgb, cls_id in RGB_TO_CLASS.items():
        class_map[np.all(mask_np == np.array(rgb, dtype=np.uint8), axis=-1)] = cls_id
    return class_map

def class_to_rgb_mask(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        rgb[class_map == cls_id] = color
    return rgb

# ─── Augmentation ────────────────────────────────────────────────────────────

def _gaussian_filter(arr, sigma):
    """2D Gaussian filter using separable 1D convolutions (numpy only)."""
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    h, w = arr.shape
    padded = np.pad(arr, radius, mode='reflect')
    row_conv = np.zeros((h + 2 * radius, w), dtype=np.float64)
    for i in range(h + 2 * radius):
        row_conv[i] = np.convolve(padded[i], kernel, mode='valid')
    result = np.zeros((h, w), dtype=np.float64)
    for j in range(w):
        result[:, j] = np.convolve(row_conv[:, j], kernel, mode='valid')
    return result

def elastic_transform(image, mask, alpha=120, sigma=12):
    """Elastic deformation - critical for histopathology."""
    shape = image.shape[:2]
    dx = _gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = _gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.clip(y + dy, 0, shape[0]-1).flatten(),
               np.clip(x + dx, 0, shape[1]-1).flatten())
    for c in range(3):
        image[:,:,c] = image[:,:,c].flatten()[np.ravel_multi_index(
            indices, shape, mode='clip')].reshape(shape)
    mask = mask.flatten()[np.ravel_multi_index(indices, shape, mode='clip')].reshape(shape)
    return image, mask

def stain_augmentation(image):
    """Simple H&E stain augmentation via channel-wise perturbation."""
    for c in range(3):
        factor = random.uniform(0.85, 1.15)
        shift = random.uniform(-10, 10)
        image[:,:,c] = np.clip(image[:,:,c].astype(np.float32) * factor + shift, 0, 255)
    return image.astype(np.uint8)

def augment_pair(image, mask):
    """Apply augmentations to image-mask pair."""
    # Spatial augmentations (applied to both)
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
    if random.random() > 0.5:
        image = np.flip(image, axis=0).copy(); mask = np.flip(mask, axis=0).copy()
    k = random.randint(0, 3)
    if k: image = np.rot90(image, k).copy(); mask = np.rot90(mask, k).copy()
    # Elastic deformation
    if random.random() > 0.5:
        image, mask = elastic_transform(image.copy(), mask.copy())
    # Color augmentations (image only)
    if random.random() > 0.5:
        image = stain_augmentation(image.copy())
    if random.random() > 0.5:
        f = random.uniform(0.7, 1.3)
        image = np.clip(image.astype(np.float32) * f, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        m = image.mean()
        f = random.uniform(0.7, 1.3)
        image = np.clip((image.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)
    if random.random() > 0.6:
        noise = np.random.normal(0, 8, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Random scale + crop
    if random.random() > 0.5:
        h, w = image.shape[:2]
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)
        img_pil = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
        msk_pil = Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
        image, mask = np.array(img_pil), np.array(msk_pil).astype(np.int64)
        if new_h >= h and new_w >= w:
            y0 = random.randint(0, new_h - h); x0 = random.randint(0, new_w - w)
            image = image[y0:y0+h, x0:x0+w]; mask = mask[y0:y0+h, x0:x0+w]
        else:
            pad_h = max(0, h - new_h); pad_w = max(0, w - new_w)
            image = np.pad(image, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')[:h,:w]
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode='reflect')[:h,:w]
    return image, mask

# ─── Dataset ─────────────────────────────────────────────────────────────────

class CoNSePDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, augment=False):
        self.image_paths, self.mask_paths, self.augment = image_paths, mask_paths, augment

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.mask_paths is not None:
            mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))
            class_map = rgb_mask_to_class(mask_rgb)
            if self.augment:
                img, class_map = augment_pair(img, class_map)
            class_map = torch.from_numpy(class_map.copy()).long()
        else:
            class_map = torch.zeros(1)
        img = torch.from_numpy(img.copy().astype(np.float32) / 255.0).permute(2,0,1)
        return img, class_map

    def get_class_weights(self):
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for mp in self.mask_paths:
            cm = rgb_mask_to_class(np.array(Image.open(mp).convert("RGB")))
            for c in range(NUM_CLASSES): counts[c] += (cm == c).sum()
        weights = counts.sum() / (NUM_CLASSES * counts + 1e-6)
        return torch.tensor(weights / weights.sum() * NUM_CLASSES, dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.tif'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(os.path.join(self.image_dir, fname)).convert("RGB"))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2,0,1)
        return img, fname

# ─── Model: Attention UNet with Residual Blocks & Deep Supervision ──────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class ResDoubleConv(nn.Module):
    """Residual double conv block with SE attention."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch))
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.relu(self.se(self.conv(x)) + self.skip(x)))

class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, int_ch, 1, bias=False), nn.BatchNorm2d(int_ch))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, int_ch, 1, bias=False), nn.BatchNorm2d(int_ch))
        self.psi = nn.Sequential(nn.Conv2d(int_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        # Handle size mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        return x * self.psi(self.relu(g1 + x1))

class AttentionUNet(nn.Module):
    """Attention UNet with residual blocks, SE, and deep supervision."""
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]
        # Encoder
        self.enc1 = ResDoubleConv(in_ch, filters[0])
        self.enc2 = ResDoubleConv(filters[0], filters[1])
        self.enc3 = ResDoubleConv(filters[1], filters[2])
        self.enc4 = ResDoubleConv(filters[2], filters[3])
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ResDoubleConv(filters[3], filters[4], dropout=0.3)
        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.ag4 = AttentionGate(filters[3], filters[3], filters[3]//2)
        self.dec4 = ResDoubleConv(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.ag3 = AttentionGate(filters[2], filters[2], filters[2]//2)
        self.dec3 = ResDoubleConv(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.ag2 = AttentionGate(filters[1], filters[1], filters[1]//2)
        self.dec2 = ResDoubleConv(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.ag1 = AttentionGate(filters[0], filters[0], filters[0]//2)
        self.dec1 = ResDoubleConv(filters[1], filters[0])

        # Deep supervision heads
        self.ds4 = nn.Conv2d(filters[3], num_classes, 1)
        self.ds3 = nn.Conv2d(filters[2], num_classes, 1)
        self.ds2 = nn.Conv2d(filters[1], num_classes, 1)
        # Main head
        self.head = nn.Conv2d(filters[0], num_classes, 1)

    def _pad_cat(self, x, skip):
        dh = skip.size(2) - x.size(2); dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        input_size = x.shape[2:]
        # Encoder
        s1 = self.enc1(x);  x = self.pool(s1)
        s2 = self.enc2(x);  x = self.pool(s2)
        s3 = self.enc3(x);  x = self.pool(s3)
        s4 = self.enc4(x);  x = self.pool(s4)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        x = self.up4(x); s4 = self.ag4(x, s4); x = self._pad_cat(x, s4); x = d4 = self.dec4(x)
        x = self.up3(x); s3 = self.ag3(x, s3); x = self._pad_cat(x, s3); x = d3 = self.dec3(x)
        x = self.up2(x); s2 = self.ag2(x, s2); x = self._pad_cat(x, s2); x = d2 = self.dec2(x)
        x = self.up1(x); s1 = self.ag1(x, s1); x = self._pad_cat(x, s1); x = self.dec1(x)
        # Outputs
        main_out = self.head(x)
        if self.training:
            ds4_out = F.interpolate(self.ds4(d4), size=input_size, mode='bilinear', align_corners=True)
            ds3_out = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=True)
            ds2_out = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=True)
            return main_out, ds4_out, ds3_out, ds2_out
        return main_out

# ─── Loss ────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        total = 0.0; cnt = 0
        for c in range(1, num_classes):  # skip background
            p, g = probs[:,c], one_hot[:,c]
            inter = (p * g).sum()
            total += (2*inter + self.smooth) / (p.sum() + g.sum() + self.smooth)
            cnt += 1
        return 1.0 - total / cnt

class DeepSupervisionLoss(nn.Module):
    """Combined Focal + Dice with deep supervision."""
    def __init__(self, class_weights=None):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=2.0)
        self.dice = DiceLoss()
        self.ds_weights = [1.0, 0.4, 0.3, 0.2]  # main, ds4, ds3, ds2

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            loss = 0
            for w, out in zip(self.ds_weights, outputs):
                loss += w * (0.5 * self.focal(out, targets) + 0.5 * self.dice(out, targets))
            return loss
        return 0.5 * self.focal(outputs, targets) + 0.5 * self.dice(outputs, targets)

# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, num_classes=NUM_CLASSES):
    ious, dices = [], []
    for c in range(1, num_classes):
        pc, tc = (preds == c), (targets == c)
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        if union > 0: ious.append((inter / union).item())
        s = pc.sum().float() + tc.sum().float()
        if s > 0: dices.append((2 * inter / s).item())
    return (np.mean(ious) if ious else 0.0, np.mean(dices) if dices else 0.0)

# ─── TTA ─────────────────────────────────────────────────────────────────────

def tta_predict(model, img, device):
    """Test-time augmentation: original + flips + rot90."""
    model.eval()
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),      # hflip
        lambda x: torch.flip(x, [2]),      # vflip
        lambda x: torch.rot90(x, 1, [2,3]), # rot90
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, -1, [2,3]),
    ]
    avg = None
    for t, inv_t in zip(transforms, inv_transforms):
        inp = t(img.unsqueeze(0).to(device))
        with torch.no_grad():
            out = model(inp)
        out = inv_t(F.softmax(out, dim=1))
        avg = out if avg is None else avg + out
    return (avg / len(transforms)).argmax(dim=1).squeeze(0)

# ─── LR Scheduler with Warmup ───────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Handle both: dataset_path = root (with train_images/) OR dataset_path/train/ with images in parent
    img_dir = os.path.join(args.dataset_path, "train_images")
    mask_dir = os.path.join(args.dataset_path, "train_masks")
    if not os.path.isdir(img_dir):
        parent = os.path.dirname(args.dataset_path)
        img_dir = os.path.join(parent, "train_images")
        mask_dir = os.path.join(parent, "train_masks")
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    img_paths = [os.path.join(img_dir, f) for f in all_imgs]
    msk_paths = [os.path.join(mask_dir, f) for f in all_imgs]

    tr_imgs, vl_imgs, tr_msks, vl_msks = train_test_split(
        img_paths, msk_paths, test_size=0.2, random_state=SEED)
    print(f"Train: {len(tr_imgs)} | Val: {len(vl_imgs)}")

    train_ds = CoNSePDataset(tr_imgs, tr_msks, augment=True)
    val_ds = CoNSePDataset(vl_imgs, vl_msks, augment=False)

    print("Computing class weights...")
    class_weights = train_ds.get_class_weights().to(device)
    print(f"Class weights: {class_weights}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = AttentionUNet(in_ch=3, num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DeepSupervisionLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    num_epochs = 150
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=num_epochs)
    scaler = GradScaler()

    best_miou = 0.0; patience_counter = 0; patience = 25

    for epoch in range(num_epochs):
        lr = scheduler.step(epoch)
        # Train
        model.train(); train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval(); val_loss = 0.0; all_miou = []; all_dice = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    logits = model(imgs)
                    loss = 0.5 * F.cross_entropy(logits, masks, weight=class_weights) + \
                           0.5 * DiceLoss()(logits, masks)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                miou, mdice = compute_metrics(preds, masks)
                all_miou.append(miou); all_dice.append(mdice)
        val_loss /= len(val_loader)
        vm, vd = np.mean(all_miou), np.mean(all_dice)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR: {lr:.6f} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"mIoU: {vm:.4f} | Dice: {vd:.4f}")

        if vm > best_miou:
            best_miou = vm; patience_counter = 0
            os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  >>> Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}"); break

    print(f"\nDone. Best mIoU: {best_miou:.4f} -> {args.model_save_path}")

# ─── Inference ───────────────────────────────────────────────────────────────

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_dir = os.path.join(args.dataset_path, "test_images")
    if not os.path.isdir(img_dir):
        parent = os.path.dirname(args.dataset_path)
        img_dir = os.path.join(parent, "test_images") if os.path.isdir(os.path.join(parent, "test_images")) else args.dataset_path

    model = AttentionUNet(in_ch=3, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded.")

    os.makedirs(args.output_path, exist_ok=True)
    test_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    print(f"Test images: {len(test_files)}")

    for fname in test_files:
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
        img_t = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        pred = tta_predict(model, img_t, device).cpu().numpy()
        rgb = class_to_rgb_mask(pred)
        base, _ = os.path.splitext(fname)
        Image.fromarray(rgb).save(os.path.join(args.output_path, base + ".png"))

    print(f"Predictions saved to: {args.output_path}")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cell Segmentation")
    parser.add_argument("--mode", type=str, required=True, choices=["train","inference"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, default="cell.pth")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="predictions")
    args = parser.parse_args()
    if args.mode == "train": train(args)
    else:
        assert args.model_path, "--model_path required for inference"
        inference(args)

if __name__ == "__main__":
    main()
