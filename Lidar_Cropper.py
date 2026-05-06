import numpy as np
import os

class KittiLidarCropper:
    """
    Cuts 3D points from LiDAR scan based on KITTI labels and saves them as .npy files.
    """
    def __init__(self, base_path, output_dir="lidar_crops"):
        self.base_path = base_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_kitti_labels(self, label_path):
        """Extracts 3D box information from KITTI label file."""
        objects = []
        with open(label_path, 'r') as f:
            for line in f:
                data = line.split()
                if data[0] == 'DontCare': continue
                objects.append({
                    'type': data[0],
                    'dims': [float(data[8]), float(data[9]), float(data[10])], # h, w, l
                    'loc': [float(data[11]), float(data[12]), float(data[13])], # x, y, z (cam)
                    'ry': float(data[14])
                })
        return objects

    def get_points_in_box(self, points, obj, V2C, R0):
        """
        Filters LiDAR points that fall inside the 3D bounding box.
        """
        # 1. Transform points from LiDAR to Camera Rectified coordinates
        pts_homo = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        r0_homo = np.eye(4); r0_homo[:3, :3] = R0
        v2c_homo = np.vstack((V2C, [0, 0, 0, 1]))
        
        # Combined transformation to Camera Rectified space
        pts_cam = (pts_homo @ v2c_homo.T @ r0_homo.T)[:, :3]

        # 2. Translate and Rotate points to Object's local coordinate system
        h, w, l = obj['dims']
        tx, ty, tz = obj['loc']
        ry = obj['ry']

        # Shift points so box center is at (0,0,0)
        # Note: In KITTI, loc is the center of the bottom face
        pts_local = pts_cam - np.array([tx, ty - h/2, tz])

        # Rotation matrix around Y axis
        R = np.array([[np.cos(-ry), 0, np.sin(-ry)],
                      [0, 1, 0],
                      [-np.sin(-ry), 0, np.cos(-ry)]])
        pts_local = pts_local @ R.T

        # 3. Filter points by box dimensions
        mask = (np.abs(pts_local[:, 0]) < l/2) & \
               (np.abs(pts_local[:, 1]) < h/2) & \
               (np.abs(pts_local[:, 2]) < w/2)
        
        return points[mask]

    def process_all(self):
        """Iterates through all training samples and saves crops."""
        lidar_dir = os.path.join(self.base_path, "velodyne")
        sample_ids = [f.split('.')[0] for f in os.listdir(lidar_dir) if f.endswith('.bin')]

        for s_id in sample_ids:
            # Load Calibration
            calib_path = os.path.join(self.base_path, "calib", f"{s_id}.txt")
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            R0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
            V2C = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)

            # Load Data
            scan = np.fromfile(os.path.join(lidar_dir, f"{s_id}.bin"), dtype=np.float32).reshape(-1, 4)
            labels = self.load_kitti_labels(os.path.join(self.base_path, "label_2", f"{s_id}.txt"))

            for i, obj in enumerate(labels):
                crop = self.get_points_in_box(scan, obj, V2C, R0)
                
                # Save even if 0 points (or filter them out)
                output_name = f"{obj['type']}_{s_id}_{i}.npy"
                np.save(os.path.join(self.output_dir, output_name), crop)
            
            print(f"Processed LiDAR for sample: {s_id}")

if __name__ == "__main__":
    BASE_PATH = r"D:\magister\coursa\kitti_root\training"
    cropper = KittiLidarCropper(BASE_PATH, output_dir=r"D:\magister\coursa\full_lidar_crops")
    cropper.process_all()