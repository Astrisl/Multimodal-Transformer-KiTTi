import torch
import numpy as np
import cv2
import os
import random
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from main import MultimodalTransformer 
from dataset_crooper import KittiDataset, KittiVisualizer

BASE_PATH = r"D:\magister\coursa\kitti_root\training"  
MODEL_PATH = "best_multimodal_model.pth"              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_TO_LABEL = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_auto_demo():
    print("Model loading...")
    model = MultimodalTransformer(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    
    sample_ids = [f.replace('.png', '') for f in os.listdir(os.path.join(BASE_PATH, "image_2")) if f.endswith('.png')]
    random_id = random.choice(sample_ids)
    print(f"Choose random frame: {random_id}")

    ds = KittiDataset(BASE_PATH, sample_id=random_id)   
    vis = KittiVisualizer(ds)
    
    target_obj = None
    for obj in ds.labels:
        if obj['type'] in ['Car', 'Pedestrian', 'Cyclist']:
            target_obj = obj
            break
            
    if target_obj is None:
        print("There are no targets in this scene, please try again..")
        return

    print(f"Object Detection: {target_obj['type']}")
    
    corners_3d, corners_2d = vis.get_3d_box_corners(target_obj)
    img = ds.get_image()
    h, w = img.shape[:2] 
    
    x1, y1 = np.clip(np.min(corners_2d, axis=0), 0, [w, h])
    x2, y2 = np.clip(np.max(corners_2d, axis=0), 0, [w, h])
    crop_img = img[int(y1):int(y2), int(x1):int(x2)]
    
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = img_transform(pil_img).unsqueeze(0).to(DEVICE) 
    
    all_points = ds.get_lidar_points()
    pts_cam = vis.project_lidar_to_cam(all_points)
    
    xmin, xmax = np.min(corners_3d[0, :]), np.max(corners_3d[0, :])
    ymin, ymax = np.min(corners_3d[1, :]), np.max(corners_3d[1, :])
    zmin, zmax = np.min(corners_3d[2, :]), np.max(corners_3d[2, :])
    mask = (pts_cam[:, 0] >= xmin) & (pts_cam[:, 0] <= xmax) & \
           (pts_cam[:, 1] >= ymin) & (pts_cam[:, 1] <= ymax) & \
           (pts_cam[:, 2] >= zmin) & (pts_cam[:, 2] <= zmax)
    
    points = all_points[mask][:, :3] 
    
    if len(points) == 0:
        points = np.zeros((1024, 3))
    elif len(points) >= 1024:
        indices = np.random.choice(len(points), 1024, replace=False)
        points = points[indices]
    else:
        indices = np.random.choice(len(points), 1024, replace=True)
        points = points[indices]
    
    pts_tensor = torch.from_numpy(points).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor, pts_tensor)
        _, predicted = torch.max(output, 1)
        label_id = predicted.item()
        final_label = ID_TO_LABEL[label_id]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Input Image Crop\n(Клас: {target_obj['type']})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.scatter(points[:, 0], points[:, 2], s=1, c='blue')
    plt.title(f"Model forecast: {final_label}")
    plt.xlabel("X (Width)")
    plt.ylabel("Z (Depth)")
    
    print(f"This is: {final_label} (Verify: {target_obj['type']})")
    plt.show()

if __name__ == "__main__":
    run_auto_demo()