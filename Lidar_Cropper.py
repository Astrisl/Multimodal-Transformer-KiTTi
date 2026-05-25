import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # Added for progress bar visualization
from typing import List, Dict, Any, Tuple

class KittiLidarCropper:
    """
    Extracts 3D points from LiDAR scans that fall strictly within the KITTI 3D bounding boxes.
    Saves the extracted object-centric point clouds as NumPy (.npy) files.
    """
    def __init__(self, base_path: str, output_dir: str = "lidar_crops"):
        """
        Initializes the cropper with dataset paths.
        
        Args:
            base_path (str): Root directory of the KITTI training dataset.
            output_dir (str): Destination directory for the cropped .npy files.
        """
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_kitti_labels(self, label_path: Path) -> List[Dict[str, Any]]:
        """
        Parses the KITTI label text file to extract object dimensions, location, and rotation.
        
        Args:
            label_path (Path): Path to the specific .txt label file.
            
        Returns:
            List[Dict]: A list of dictionaries containing object metadata.
        """
        objects = []
        if not label_path.exists():
            return objects

        with open(label_path, 'r') as f:
            for line in f:
                data = line.split()
                # Skip 'DontCare' or 'Misc' regions as they are not valid objects
                if data[0] in ['DontCare', 'Misc']: 
                    continue
                
                objects.append({
                    'type': data[0],
                    'dims': [float(data[8]), float(data[9]), float(data[10])], # [height, width, length]
                    'loc': [float(data[11]), float(data[12]), float(data[13])], # [x, y, z] in camera coords
                    'ry': float(data[14]) # Rotation around Y-axis in camera coords
                })
        return objects

    def get_points_in_box(self, points: np.ndarray, obj: Dict[str, Any], V2C: np.ndarray, R0: np.ndarray) -> np.ndarray:
        """
        Applies geometric transformations to filter LiDAR points inside a 3D bounding box.
        
        Args:
            points (np.ndarray): Raw LiDAR point cloud [N, 4] (x, y, z, reflectance).
            obj (Dict): Parsed object metadata.
            V2C (np.ndarray): Velodyne-to-Camera transformation matrix [3, 4].
            R0 (np.ndarray): Camera rectification matrix [3, 3].
            
        Returns:
            np.ndarray: Filtered point cloud containing only points inside the object box.
        """
        # 1. Transform points from LiDAR to Camera Rectified coordinate system
        pts_3d = points[:, :3]
        pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        
        r0_homo = np.eye(4)
        r0_homo[:3, :3] = R0
        v2c_homo = np.vstack((V2C, [0, 0, 0, 1]))
        
        # Combined transformation: X_cam = R0 * V2C * X_velo
        pts_cam = (pts_homo @ v2c_homo.T @ r0_homo.T)[:, :3]

        # 2. Translate and Rotate points to the Object's local coordinate system
        h, w, l = obj['dims']
        tx, ty, tz = obj['loc']
        ry = obj['ry']

        # Shift points so the bounding box center is at origin (0,0,0)
        # Note: KITTI location 'ty' is the bottom face of the object, so we shift by h/2
        pts_local = pts_cam - np.array([tx, ty - h/2, tz])

        # Apply inverse rotation around the Y-axis to axis-align the bounding box
        R_inv = np.array([
            [np.cos(-ry), 0, np.sin(-ry)],
            [0,          1,          0],
            [-np.sin(-ry), 0, np.cos(-ry)]
        ])
        pts_local = pts_local @ R_inv.T

        # 3. Filter points using strict axis-aligned spatial boundaries
        mask = (np.abs(pts_local[:, 0]) <= l/2) & \
               (np.abs(pts_local[:, 1]) <= h/2) & \
               (np.abs(pts_local[:, 2]) <= w/2)
               
        # Return original points (including reflectance) that satisfy the mask
        return points[mask]

    def process_all(self):
        """
        Iterates through the entire training dataset, crops objects, and saves them.
        Uses TQDM for a visual progress bar.
        """
        lidar_dir = self.base_path / "velodyne"
        if not lidar_dir.exists():
            raise FileNotFoundError(f"LiDAR directory not found at {lidar_dir}")

        sample_ids = sorted([f.stem for f in lidar_dir.glob('*.bin')])
        
        empty_crops_count = 0 # Track how many objects had 0 LiDAR points

        # Using tqdm instead of print() for performance and better UX
        for s_id in tqdm(sample_ids, desc="Processing LiDAR Scans"):
            
            # Load Calibration matrices
            calib_path = self.base_path / "calib" / f"{s_id}.txt"
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            R0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
            V2C = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)

            # Load LiDAR binary data and Labels
            scan_path = lidar_dir / f"{s_id}.bin"
            scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            
            label_path = self.base_path / "label_2" / f"{s_id}.txt"
            labels = self.load_kitti_labels(label_path)

            # Process each valid object in the scene
            for i, obj in enumerate(labels):
                crop = self.get_points_in_box(scan, obj, V2C, R0)
                
                # Improvement: Skip saving completely empty point clouds to save space
                # and prevent DataLoader crashes during training.
                if crop.shape[0] == 0:
                    empty_crops_count += 1
                    continue
                
                output_name = f"{obj['type']}_{s_id}_{i}.npy"
                np.save(self.output_dir / output_name, crop)
                
        print(f"\nProcessing complete! Saved to {self.output_dir}")
        print(f"Skipped {empty_crops_count} objects due to 0 LiDAR points (occluded or too far).")

if __name__ == "__main__":
    # Define paths (Use raw strings 'r' to prevent escape character issues in Windows)
    BASE_PATH = r"D:\magister\coursa\kitti_root\training"
    OUTPUT_PATH = r"D:\magister\coursa\full_lidar_crops"
    
    cropper = KittiLidarCropper(base_path=BASE_PATH, output_dir=OUTPUT_PATH)
    cropper.process_all()