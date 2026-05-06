import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class KittiDataset:
    """Class for work with KITTI data:download and transforming"""
    def __init__(self, base_path, sample_id="000000"):
        self.sample_id = sample_id
        self.img_path = os.path.join(base_path, "image_2", f"{sample_id}.png")
        self.lidar_path = os.path.join(base_path, "velodyne", f"{sample_id}.bin")
        self.calib_path = os.path.join(base_path, "calib", f"{sample_id}.txt")
        self.label_path = os.path.join(base_path, "label_2", f"{sample_id}.txt")
        
        self.P2, self.R0, self.V2C = self._load_calib()
        self.labels = self._load_labels()

    def _load_calib(self):
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
        p2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        r0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
        v2c = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
        return p2, r0, v2c

    def _load_labels(self):
        objects = []
        if not os.path.exists(self.label_path): return objects
        with open(self.label_path, 'r') as f:
            for line in f:
                data = line.split()
                objects.append({
                    'type': data[0],
                    'dimensions': [float(data[8]), float(data[9]), float(data[10])], # h, w, l
                    'location': [float(data[11]), float(data[12]), float(data[13])], # x, y, z
                    'rotation_y': float(data[14])
                })
        return objects

    def get_lidar_points(self):
        return np.fromfile(self.lidar_path, dtype=np.float32).reshape(-1, 4)

    def get_image(self):
        return cv2.imread(self.img_path)


class KittiVisualizer:
    """Class for visualisation and preproccesing"""
    def __init__(self, dataset: KittiDataset):
        self.ds = dataset

    def project_lidar_to_2d(self, points):
        points = points[points[:, 0] > 0]
        pts_3d = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))

        r0_homo = np.eye(4); r0_homo[:3, :3] = self.ds.R0
        v2c_homo = np.vstack((self.ds.V2C, [0, 0, 0, 1]))
        
        # Проєкція
        pts_2d = pts_3d @ v2c_homo.T @ r0_homo.T @ self.ds.P2.T
        depths = pts_2d[:, 2]
        pts_2d[:, :2] /= pts_2d[:, 2:3]
        
        return pts_2d[:, :2], depths

    def get_3d_box_corners(self, obj):
        h, w, l = obj['dimensions']
        ry = obj['rotation_y']
        R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        
        corners_3d = R @ np.vstack([x_corners, y_corners, z_corners])
        corners_3d += np.array(obj['location']).reshape(3, 1)
        
        corners_3d_homo = np.vstack((corners_3d, np.ones((1, 8))))
        pts_2d = self.ds.P2 @ corners_3d_homo
        pts_2d[:2] /= pts_2d[2, :]
        return pts_2d[:2, :].T.astype(np.int32)

    def auto_crop(self, output_dir="crops", padding=5):
        img = self.ds.get_image()
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        for i, obj in enumerate(self.ds.labels):
            if obj['type'] == 'DontCare': continue
            corners = self.get_3d_box_corners(obj)
            x1, y1 = np.clip(np.min(corners, axis=0) - padding, 0, [img.shape[1], img.shape[0]])
            x2, y2 = np.clip(np.max(corners, axis=0) + padding, 0, [img.shape[1], img.shape[0]])

            if x2 > x1 and y2 > y1:
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                cv2.imwrite(os.path.join(output_dir, f"{obj['type']}_{self.ds.sample_id}_{i}.png"), crop)

    def show_projection(self):
        img = self.ds.get_image()
        points = self.ds.get_lidar_points()
        pts_2d, depths = self.project_lidar_to_2d(points)
        
        h, w, _ = img.shape
        mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        pts_2d, depths = pts_2d[mask], depths[mask]

        for i in range(len(pts_2d)):
            color = plt.get_cmap('jet')(depths[i] / 40.0)[:3]
            cv2.circle(img, (int(pts_2d[i, 0]), int(pts_2d[i, 1])), 1, tuple(int(c*255) for c in color[::-1]), -1)
        
        cv2.imshow("Kitti OOP Visualizer", img)
        cv2.waitKey(0)

class KittiFullProcessor(KittiVisualizer):
    def __init__(self, base_path):
        self.base_path = base_path
        self.lidar_dir = os.path.join(base_path, "velodyne")
        self.sample_ids = [f.split('.')[0] for f in os.listdir(self.lidar_dir) if f.endswith('.bin')]
        print(f"Found {len(self.sample_ids)} process frame.")

    def run_full_extraction(self, output_dir="dataset_crops"):
        for s_id in self.sample_ids:
            try:
                current_ds = KittiDataset(self.base_path, sample_id=s_id)
                self.ds = current_ds

                self.auto_crop(output_dir=output_dir)
                print(f"Refactor frame: {s_id}")
            except Exception as e:
                print(f"Error in frame: {s_id}: {e}")




if __name__ == "__main__":
    BASE_PATH = r"D:\magister\coursa\kitti_root\training"

    processor = KittiFullProcessor(BASE_PATH)
    processor.run_full_extraction(output_dir=r"D:\magister\coursa\full_dataset_crops")
    
    # dataset = KittiDataset(BASE_PATH, sample_id="000005")
    
    # vis = KittiVisualizer(dataset)

    # vis.auto_crop()
    # vis.show_projection()