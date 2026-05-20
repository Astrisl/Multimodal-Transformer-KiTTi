import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

class KittiDataset:
    """Class for loading, parsing, and coordinate transformations of KITTI dataset data."""
    def __init__(self, base_path, sample_id="000000"):
        self.sample_id = sample_id
        self.img_path = os.path.join(base_path, "image_2", f"{sample_id}.png")
        self.lidar_path = os.path.join(base_path, "velodyne", f"{sample_id}.bin")
        self.calib_path = os.path.join(base_path, "calib", f"{sample_id}.txt")
        self.label_path = os.path.join(base_path, "label_2", f"{sample_id}.txt")
        
        self.P2, self.R0, self.V2C = self._load_calib()
        self.labels = self._load_labels()

    def _load_calib(self):
        """Read calibration matrices according to the KITTI specification."""
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
        
        # P2: Projection matrix from rectified camera 0 coordinate system to 2D image of camera 2 (3x4)
        p2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        # R0_rect: Rectification matrix for camera 0 (3x3)
        r0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
        # Tr_velo_to_cam: Transformation from LiDAR to camera 0 coordinate system (3x4)
        v2c = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
        return p2, r0, v2c

    def _load_labels(self):
        """Parsing object annotation labels (Ground Truth)."""
        objects = []
        if not os.path.exists(self.label_path): 
            return objects
        with open(self.label_path, 'r') as f:
            for line in f:
                data = line.split()
                objects.append({
                    'type': data[0],
                    'dimensions': [float(data[8]), float(data[9]), float(data[10])], # h, w, l
                    'location': [float(data[11]), float(data[12]), float(data[13])], # x, y, z (in camera system)
                    'rotation_y': float(data[14])
                })
        return objects

    def get_lidar_points(self):
        """Load binary point cloud file (x, y, z, r)."""
        return np.fromfile(self.lidar_path, dtype=np.float32).reshape(-1, 4)

    def get_image(self):
        """Read frontal RGB image."""
        return cv2.imread(self.img_path)


class KittiVisualizer:
    """Class for processing, projecting sensor data, and generating multimodal crops."""
    def __init__(self, dataset: KittiDataset):
        self.ds = dataset

    def project_lidar_to_cam(self, points):
        """Convert LiDAR points to the 3D rectified camera 0 coordinate system."""
        pts_3d = points[:, :3]
        # 1. From LiDAR to Rectified Cam using official KITTI formula: Y = R0_rect * T_velo_to_cam * X
        pts_cam = (self.ds.R0 @ (self.ds.V2C[:, :3] @ pts_3d.T + self.ds.V2C[:, 3:4])).T
        return pts_cam

    def project_lidar_to_2d(self, points):
        """Project LiDAR points directly onto the 2D image pixel plane."""
        # Filter out points behind the vehicle (x <= 0 in the LiDAR system)
        points_filtered = points[points[:, 0] > 0]
        
        # Convert to camera 3D space
        pts_cam = self.project_lidar_to_cam(points_filtered)
        
        # Add homogeneous coordinate for multiplying by P2
        pts_cam_homo = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_2d = (self.ds.P2 @ pts_cam_homo.T).T
        
        depths = pts_2d[:, 2]
        pts_2d[:, :2] /= pts_2d[:, 2:3] # Normalize by Z (depth)
        
        return pts_2d[:, :2], depths

    def get_3d_box_corners(self, obj):
        """Compute the 8 corners of the 3D bounding box in the camera coordinate system."""
        h, w, l = obj['dimensions']
        ry = obj['rotation_y']
        
        # Rotation matrix around the vertical Y-axis (camera coordinate system)
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)], 
            [0, 1, 0], 
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Corner coordinates relative to the object center
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        corners_3d = R @ np.vstack([x_corners, y_corners, z_corners])
        corners_3d += np.array(obj['location']).reshape(3, 1) # Shift by the position vector
        
        # Project 3D box corners onto the 2D image plane
        corners_3d_homo = np.vstack((corners_3d, np.ones((1, 8))))
        pts_2d = self.ds.P2 @ corners_3d_homo
        pts_2d[:2] /= pts_2d[2, :]
        return corners_3d, pts_2d[:2, :].T.astype(np.int32)

    def auto_crop(self, output_img_dir="crops_img", output_lidar_dir="crops_lidar", padding=5):
        """Automatically generate and synchronize image crops and LiDAR points crops."""
        img = self.ds.get_image()
        lidar_points = self.ds.get_lidar_points()
        
        # Convert all LiDAR points to camera 3D space for geometric filtering inside the box
        pts_cam = self.project_lidar_to_cam(lidar_points)

        if not os.path.exists(output_img_dir): os.makedirs(output_img_dir)
        if not os.path.exists(output_lidar_dir): os.makedirs(output_lidar_dir)

        for i, obj in enumerate(self.ds.labels):
            if obj['type'] in ['DontCare', 'Misc']: 
                continue
                
            corners_3d, corners_2d = self.get_3d_box_corners(obj)
            
            # --- 1. Crop RGB image ---
            x1, y1 = np.clip(np.min(corners_2d, axis=0) - padding, 0, [img.shape[1], img.shape[0]])
            x2, y2 = np.clip(np.max(corners_2d, axis=0) + padding, 0, [img.shape[1], img.shape[0]])

            if x2 > x1 and y2 > y1:
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                img_name = f"{obj['type']}_{self.ds.sample_id}_{i}.png"
                cv2.imwrite(os.path.join(output_img_dir, img_name), crop)

                # --- 2. Crop LiDAR points (Geometric filtering using 3D Bounding Box) ---
                # Determine object boundaries in camera space
                xmin, xmax = np.min(corners_3d[0, :]), np.max(corners_3d[0, :])
                ymin, ymax = np.min(corners_3d[1, :]), np.max(corners_3d[1, :])
                zmin, zmax = np.min(corners_3d[2, :]), np.max(corners_3d[2, :])
                
                # Mask for points falling inside the 3D bounding box
                mask = (pts_cam[:, 0] >= xmin) & (pts_cam[:, 0] <= xmax) & \
                       (pts_cam[:, 1] >= ymin) & (pts_cam[:, 1] <= ymax) & \
                       (pts_cam[:, 2] >= zmin) & (pts_cam[:, 2] <= zmax)
                
                # Save raw LiDAR points that fell into the object zone
                crop_points = lidar_points[mask]
                
                lidar_name = f"{obj['type']}_{self.ds.sample_id}_{i}.bin"
                crop_points.tofile(os.path.join(output_lidar_dir, lidar_name))


class KittiFullProcessor(KittiVisualizer):
    """Class for batch processing the entire sequential KITTI dataset."""
    def __init__(self, base_path):
        self.base_path = base_path
        self.lidar_dir = os.path.join(base_path, "velodyne")
        if not os.path.exists(self.lidar_dir):
            raise FileNotFoundError(f"Directory {self.lidar_dir} not found! Check BASE_PATH.")
        self.sample_ids = sorted([f.split('.')[0] for f in os.listdir(self.lidar_dir) if f.endswith('.bin')])
        print(f"[INFO] Found {len(self.sample_ids)} frames for dataset generation.")

    def run_full_extraction(self, base_output_dir):
        """Run the full feature extraction pipeline."""
        img_out = os.path.join(base_output_dir, "crop_images")
        lidar_out = os.path.join(base_output_dir, "crop_lidar")
        
        for idx, s_id in enumerate(self.sample_ids):
            try:
                current_ds = KittiDataset(self.base_path, sample_id=s_id)
                self.ds = current_ds
                
                # Run synchronized sensor cropping
                self.auto_crop(output_img_dir=img_out, output_lidar_dir=lidar_out)
                
                if (idx + 1) % 50 == 0 or (idx + 1) == len(self.sample_ids):
                    print(f"[PROGRESS] Processed frames: {idx + 1}/{len(self.sample_ids)}")
            except Exception as e:
                print(f"[ERROR] Error processing frame {s_id}: {e}")


if __name__ == "__main__":
    # Configure CLI arguments (best practice instead of hardcoding paths)
    parser = argparse.ArgumentParser(description="KITTI Multimodal Preprocessing Pipeline")
    parser.add_argument('--base_path', type=str, default=r"D:\magister\coursa\kitti_root\training", 
                        help="Path to the original training folder of the KITTI dataset")
    parser.add_argument('--output_path', type=str, default=r"D:\magister\coursa\full_dataset_crops", 
                        help="Path to save the resulting synchronized multimodal crops")
    args = parser.parse_args()

    print("=== Starting multimodal data preparation ===")
    processor = KittiFullProcessor(args.base_path)
    processor.run_full_extraction(base_output_dir=args.output_path)
    print("=== Processing completed successfully! ===")