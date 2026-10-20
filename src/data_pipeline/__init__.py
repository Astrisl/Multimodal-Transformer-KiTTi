"""Turns raw KITTI / live sensor data into model-ready tensors.

Calibration, LiDAR cropping by oriented boxes, point cloud normalisation, image ROI
cropping and live ROI extraction (clustering + OBB fitting). The same code is used by
offline training and by the real-time ROS 2 pipeline.
"""
