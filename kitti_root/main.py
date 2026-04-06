import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt 
import os

def visualize_lidar(bin_path):
    if not os.path.exists(bin_path):
        print(f"Файл не знайдено: {bin_path}")
        return

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    intensities = points[:, 3]
    if len(intensities) > 0:
        norm_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        colors = plt.get_cmap("jet")(norm_intensities)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("Візуалізація запущена... Закрийте вікно Open3D, щоб продовжити.")
    o3d.visualization.draw_geometries([pcd])

visualize_lidar(r"D:\magister\coursa\kitti_root\training\velodyne\000000.bin")