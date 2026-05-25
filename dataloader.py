import enum
import torch
import time
import os
import json

import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tkinter.ttk import Progressbar
from tqdm import tqdm
from PIL import Image

class KittiMultimodalDataset(Dataset):
    def __init__(self, crop_dir, lidar_crop_dir, img_transform=None, num_points=1024, mode='train', split_ratio=0.8):
        self.crop_dir = crop_dir
        self.lidar_crop_dir = lidar_crop_dir
        self.img_transform = img_transform
        self.num_points = num_points
        self.label_map = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Truck': 0, 'Van': 0}
    
        all_files = [f for f in os.listdir(crop_dir) if f.endswith('.png')]
        unique_scenes = list(set(f.split('_')[1] for f in all_files))
        unique_scenes.sort()
    
        np.random.seed(42)
        np.random.shuffle(unique_scenes)
    
        split_idx = int(len(unique_scenes) * split_ratio)
        if mode == 'train':
            allowed_scenes = set(unique_scenes[:split_idx])
        else:
            allowed_scenes = set(unique_scenes[split_idx:])
        
        self.file_list = [f for f in all_files if f.split('_')[1] in allowed_scenes]

    def __len__(self):
        return len(self.file_list)

    def _sample_points(self, points):
        if len(points) == 0: return np.zeros((self.num_points, 3))
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.crop_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.img_transform: image = self.img_transform(image)

        npy_name = img_name.replace('.png', '.npy')
        npy_path = os.path.join(self.lidar_crop_dir, npy_name)
        
        if os.path.exists(npy_path):
            raw_points = np.load(npy_path)
            points = self._sample_points(raw_points[:, :3] if raw_points.ndim > 1 else np.zeros((1,3)))
        else:
            points = np.zeros((self.num_points, 3))

        label = self.label_map.get(img_name.split('_')[0], 0)
        return {'image': image, 'points': torch.from_numpy(points).float(), 'label': torch.tensor(label, dtype=torch.long)}

# --- 2. Architecture ---
class MultimodalTransformer(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128, nhead=8, num_layers=3):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # delete last layers
        self.cnn_extractor= nn.Sequential(*list(resnet.children())[:-2])
        
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False
        
        self.image_projection = nn.Linear(512, embed_dim) #512 for 7x7 map
        self.point_projection = nn.Linear(3, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, image_tensor, point_cloud):
        # image_tensor: [B, 3, 224, 224]
        with torch.no_grad():
            img_features = self.cnn_extractor(image_tensor) # [B, 512, 7, 7]
        
        # [B, 512, 7, 7] -> [B, 512, 49] -> [B, 49, 512]
        img_features = img_features.flatten(2).permute(0, 2, 1)
        img_embed = self.image_projection(img_features) # [B, 49, embed_dim]
        
        # Points: [B, 1024, 3] -> [B, 1024, embed_dim]
        point_embed = self.point_projection(point_cloud) 
        
        x = torch.cat((img_embed, point_embed), dim=1) # [B, 1073, embed_dim]
        
        x = self.transformer_encoder(x)
        
        return self.classifier(torch.mean(x, dim=1))

# --- 3. Trainer with Validation ---
class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_acc = 0.0
        self.history = []

    def train(self, epochs=10):
        print(f"Train started on device: {self.device}")

        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            total_loss, correct, total = 0, 0, 0
        
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
            for batch_idx, batch in pbar:
                img = batch['image'].to(self.device)
                pts = batch['points'].to(self.device)
                lbl = batch['label'].to(self.device)
            
                self.optimizer.zero_grad()
                out = self.model(img, pts)
            
                loss = self.criterion(out, lbl)
                loss.backward()
                self.optimizer.step()
            
                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()

                if batch_idx % 10 == 0:
                    current_acc = 100.0 * correct / total
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}", 
                        'Acc': f"{current_acc:.2f}%"
                    })

            epoch_loss = total_loss / len(self.train_loader)
            train_acc = 100.0 * correct / total
            val_acc = self.validate()
            epoch_duration = time.time() - start_time
        
            epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "duration_sec": round(epoch_duration, 2)
            }
            self.history.append(epoch_log)

            print(f"\n📊 [Epoch {epoch+1}] Time: {epoch_duration:.1f}с | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_multimodal_model.pth")
                print(f"⭐ New validation record.")

            with open("training_history.json", "w") as f:
                json.dump(self.history, f)
                    
    def validate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                img, pts, lbl = batch['image'].to(self.device), batch['points'].to(self.device), batch['label'].to(self.device)
                out = self.model(img, pts)
                _, pred = torch.max(out, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
        return 100 * correct / total

# --- 4. Main ---
if __name__ == "__main__":
    CROP_DIR = r"D:\magister\coursa\full_dataset_crops"
    LIDAR_DIR = r"D:\magister\coursa\full_lidar_crops"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {DEVICE}")

    img_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset and divide 80/20
    full_ds = KittiMultimodalDataset(CROP_DIR, LIDAR_DIR, img_transform=img_pipeline)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    
    train_ds = KittiMultimodalDataset(CROP_DIR, LIDAR_DIR, img_transform=img_pipeline, mode='train')
    val_ds = KittiMultimodalDataset(CROP_DIR, LIDAR_DIR, img_transform=img_pipeline, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = MultimodalTransformer(num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Class mass: Car=1, Pedestrian=5, Cyclist=10 
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 10.0]).to(DEVICE))

    trainer = MultimodalTrainer(model, train_loader, val_loader, criterion, optimizer, DEVICE)
    trainer.train(epochs=10)
    torch.save(model.state_dict(), "multimodal_transformer_final.pth")