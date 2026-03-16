import numpy as np
import cv2

def load_velo_scan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    return scan.reshape((-1, 4))

def fuse_lidar_with_thermal(lidar_points, thermal_img, K, Tr_velo_to_cam):
    img_height, img_width = thermal_img.shape
    
    points_3d = lidar_points[:, :3]
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    points_cam = (Tr_velo_to_cam @ points_hom.T).T
    
    front_idx = points_cam[:, 2] > 0
    points_cam = points_cam[front_idx]
    original_idx = np.arange(lidar_points.shape[0])[front_idx] # save original indices for valid points
    
    points_2d_hom = (K @ points_cam[:, :3].T).T
    
    u = (points_2d_hom[:, 0] / points_2d_hom[:, 2]).astype(int)
    v = (points_2d_hom[:, 1] / points_2d_hom[:, 2]).astype(int)
    
    valid_idx = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    
    u_valid = u[valid_idx]
    v_valid = v[valid_idx]
    final_3d_idx = original_idx[valid_idx]
    
    temperatures = thermal_img[v_valid, u_valid]
    
    fused_points = np.zeros((len(final_3d_idx), 4))
    fused_points[:, :3] = lidar_points[final_3d_idx, :3]
    fused_points[:, 3] = temperatures
    
    return fused_points

if __name__ == "__main__":
    lidar_file = "path/to/kaist/lidar/000000.bin" 
    thermal_file = "path/to/kaist/lwir/000000.png"
    
    try:
        lidar_data = load_velo_scan(lidar_file)
        thermal_image = cv2.imread(thermal_file, cv2.IMREAD_GRAYSCALE) 
    except FileNotFoundError:
        print("files not found, using test data.")
        lidar_data = np.random.rand(10000, 4) * 10 
        thermal_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    K_thermal = np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0]
    ])
    
    Tr_lidar_to_thermal = np.array([
        [ 0.0, -1.0,  0.0,  0.0], # Rotation
        [ 0.0,  0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0, -0.1], 
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    fused_cloud = fuse_lidar_with_thermal(lidar_data, thermal_image, K_thermal, Tr_lidar_to_thermal)
    
    print(f"First number of lidar points: {len(lidar_data)}")
    print(f"Number of points with assigned temperature: {len(fused_cloud)}")
    print("First 3 points (X, Y, Z, Temp):")
    print(fused_cloud[:3])