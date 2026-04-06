import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def point_cloud_to_bev(points, res=0.1, side_range=(-20, 20), fwd_range=(0, 40)):
    """
    Converts LiDAR point cloud to Bird's Eye View 2D image.
    res: resolution (meters per pixel)
    side_range: left/right range from the car (meters)
    fwd_range: forward range from the car (meters)
    """
    # 1. Filter points based on the defined 2D rectangle (ROI)
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter_mask = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter_mask).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # 2. Map metric coordinates to pixel coordinates
    # In KITTI: X is Forward, Y is Left/Right
    x_img = (-y_points / res).astype(np.int32) 
    y_img = (-x_points / res).astype(np.int32)

    # 3. Shift coordinates so the minimum value is 0 (centering)
    x_img -= int(side_range[0] / res)
    y_img -= int(fwd_range[0] / res)

    # 4. Initialize empty BEV image (Grid)
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    bev_img = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)

    # 5. Fill pixels with normalized Height (Z) values
    z_min, z_max = -2, 2 # Typical height range from road to roof
    pixel_values = np.clip(z_points, z_min, z_max)
    pixel_values = (((pixel_values - z_min) / (z_max - z_min)) * 255).astype(np.uint8)

    # Note: OpenCV uses (row, col), so we use bev_img[y, x]
    bev_img[y_img, x_img] = pixel_values

    return bev_img

def load_kitti_labels(label_path):
    """Parses KITTI label file into a list of dictionaries."""
    objects = []
    if not os.path.exists(label_path):
        return objects
    with open(label_path, 'r') as f:
        for line in f:
            data = line.split()
            objects.append({
                'type': data[0],
                'dimensions': [float(data[8]), float(data[9]), float(data[10])], # h, w, l
                'location': [float(data[11]), float(data[12]), float(data[13])], # x, y, z (Camera)
                'rotation_y': float(data[14])
            })
    return objects

# --- 2. BEV with Boxes Logic ---
def draw_bev_boxes(bev_img, labels, res=0.1, side_range=(-20, 20), fwd_range=(0, 40)):
    """Draws 3D bounding boxes projected onto the 2D BEV map."""
    # Convert grayscale to color to see green boxes
    bev_with_boxes = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)
    
    for obj in labels:
        if obj['type'] == 'DontCare': continue
        
        # Extract coordinates (Camera System: X=right, Y=down, Z=forward)
        c_x, c_y, c_z = obj['location']
        h, w, l = obj['dimensions']
        
        # Convert to LiDAR System: X=forward, Y=left, Z=up
        # Note: We use approximate conversion if you haven't loaded calib yet
        l_x, l_y = c_z, -c_x 
        
        # Map to BEV pixels
        # Forward (l_x) maps to Y, Side (l_y) maps to X
        pixel_x = int((-l_y - side_range[0]) / res)
        pixel_y = int((fwd_range[1] - l_x) / res) # Flip Y for top-down view
        
        # Box size in pixels
        p_w, p_l = int(w / res), int(l / res)
        
        # Draw the rectangle
        top_left = (pixel_x - p_w//2, pixel_y - p_l//2)
        bottom_right = (pixel_x + p_w//2, pixel_y + p_l//2)
        
        color = (0, 255, 0) if obj['type'] == 'Car' else (0, 255, 255)
        cv2.rectangle(bev_with_boxes, top_left, bottom_right, color, 1)
        cv2.putText(bev_with_boxes, obj['type'], (top_left[0], top_left[1]-2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    return bev_with_boxes

def get_calib_matrices(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    def parse_line(line):
        return np.array([float(x) for x in line.split()[1:]])

    # P2 is 3x4, R0 is 3x3, Tr_velo_to_cam is 3x4
    P2 = parse_line(lines[2]).reshape(3, 4)
    R0 = parse_line(lines[4]).reshape(3, 3)
    V2C = parse_line(lines[5]).reshape(3, 4)
    
    return P2, R0, V2C

def load_calib(calib_path):
    """Parses KITTI calibration file."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
    R0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    V2C = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)
    return P2, R0, V2C

def project_lidar_to_image(img, points, P2, R0, V2C):
    """Projects 3D LiDAR points onto 2D image plane."""
    # 1. Filter points that are behind the camera (x < 0)
    points = points[points[:, 0] > 0]
    
    # 2. Convert to homogeneous coordinates [x, y, z, 1]
    pts_3d = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    
    # 3. Transform: LiDAR -> Camera Gray -> Camera Rectified
    # Formula: P2 * R0_rect * Tr_velo_to_cam * P_velo
    R0_homo = np.eye(4)
    R0_homo[:3, :3] = R0
    V2C_homo = np.vstack((V2C, [0, 0, 0, 1]))
    
    # Combined transformation matrix
    pts_2d = pts_3d @ V2C_homo.T @ R0_homo.T @ P2.T
    
    # 4. Project to 2D (divide by depth Z)
    depths = pts_2d[:, 2]
    pts_2d[:, 0] /= depths
    pts_2d[:, 1] /= depths
    
    # 5. Filter points within image boundaries
    img_h, img_w, _ = img.shape
    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & \
           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    
    return pts_2d[mask, :2], depths[mask]


def project_3d_box_to_2d(obj, P2, R0, V2C):
    """
    Projects 3D bounding box corners to 2D image plane.
    """
    # 1. Get 3D corners in camera coordinates
    h, w, l = obj['dimensions']
    tx, ty, tz = obj['location']
    ry = obj['rotation_y']
    
    # Rotation matrix (around Y axis)
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    
    # Define 8 corners of the box
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    corners_3d += np.array([tx, ty, tz]).reshape(3, 1)
    
    # 2. Project to 2D using P2 and R0 (since we are already in camera coords)
    corners_3d_homo = np.vstack((corners_3d, np.ones((1, 8))))
    pts_2d = np.dot(P2, corners_3d_homo)
    pts_2d[:2] /= pts_2d[2, :]
    
    return pts_2d[:2, :].T.astype(np.int32)

def crop_objects(img, labels, P2, R0, V2C, output_dir="crops"):
    """
    Projects 3D boxes to 2D, crops them from the image, and saves to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, obj in enumerate(labels):
        if obj['type'] == 'DontCare': continue
        
        # 1. Project 3D box corners to 2D pixels
        corners_2d = project_3d_box_to_2d(obj, P2, R0, V2C)
        
        # 2. Find the bounding box of the projected corners
        x_min, y_min = np.min(corners_2d, axis=0)
        x_max, y_max = np.max(corners_2d, axis=0)
        
        # 3. Add some padding and clip to image boundaries
        padding = 5
        h, w, _ = img.shape
        x1, y1 = max(0, x_min - padding), max(0, y_min - padding)
        x2, y2 = min(w, x_max + padding), min(h, y_max + padding)
        
        # 4. Crop and Save
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            file_name = f"{obj['type']}_{i:03d}.png"
            cv2.imwrite(os.path.join(output_dir, file_name), crop)
            print(f"Saved: {file_name}")


# --- Main Execution ---
img_path = r"D:\magister\coursa\kitti_root\training\image_2\000000.png"
bin_path = r"D:\magister\coursa\kitti_root\training\velodyne\000000.bin"
calib_path = r"D:\magister\coursa\kitti_root\training\calib\000000.txt"
label_path = r"D:\magister\coursa\kitti_root\training\label_2\000000.txt"

# Check if all necessary files exist
if all(os.path.exists(p) for p in [img_path, bin_path, calib_path, label_path]):
    # 1. Load all data
    img = cv2.imread(img_path)
    img_display = img.copy() # Copy for visualization so crops remain clean
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    P2, R0, V2C = load_calib(calib_path)
    labels = load_kitti_labels(label_path)
    
    # 2. Project LiDAR points to image for visualization
    pts_2d, depths = project_lidar_to_image(img_display, scan, P2, R0, V2C)
    
    print(f"Projecting {len(pts_2d)} points onto the image...")
    for i in range(len(pts_2d)):
        color = plt.get_cmap('jet')(depths[i] / 40.0)[:3]
        color = tuple([int(c * 255) for c in color[::-1]]) 
        cv2.circle(img_display, (int(pts_2d[i, 0]), int(pts_2d[i, 1])), 1, color, -1)
    
    # 3. RUN THE CROPPER
    print("Starting to crop objects...")
    crop_objects(img, labels, P2, R0, V2C, output_dir="crops")
    print("Done! Check the 'crops' folder in your directory.")

    # 4. Show the visual result
    cv2.imshow("LiDAR Projective Fusion & Cropping", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: One or more files (including labels) are missing!")