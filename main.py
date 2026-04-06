import numpy as np
import cv2
import os

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

bin_path = r"D:\magister\coursa\kitti_root\training\velodyne\000000.bin"
label_path = r"D:\magister\coursa\kitti_root\training\label_2\000000.txt"

if os.path.exists(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    labels = load_kitti_labels(label_path)
    
    # Generate base BEV
    # point_cloud_to_bev is your function from the previous step
    bev_map = point_cloud_to_bev(scan) 
    
    # Add boxes
    final_view = draw_bev_boxes(bev_map, labels)
    
    cv2.imshow("BEV Object Detection View", final_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()