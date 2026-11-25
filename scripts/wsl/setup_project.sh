#!/usr/bin/env bash
# Step 2 of 3: project venv (sees ROS 2's rclpy through --system-site-packages), CUDA
# PyTorch, an editable install of this checkout, and a copy of the KITTI raw drives on
# the Linux file system (reading thousands of scans from /mnt/d over 9P is slow).
#   wsl -d Ubuntu-22.04 -u root -- bash scripts/wsl/setup_project.sh
set -eo pipefail
source /opt/ros/humble/setup.bash
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-$HOME/mmt-venv}
RAW_COPY=${RAW_COPY:-$HOME/kitti_raw}

python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --no-cache-dir --upgrade pip
"$VENV/bin/pip" install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
# numpy<2 and opencv<4.12 stay ABI-compatible with ROS 2 Humble's cv_bridge.
"$VENV/bin/pip" install --no-cache-dir "numpy<2" scipy "opencv-python-headless<4.12" \
    pillow pyyaml tqdm matplotlib
"$VENV/bin/pip" install --no-cache-dir --no-deps -e "$REPO"

for date_dir in "$REPO"/data/kitti_raw/*/; do
  date=$(basename "$date_dir")
  mkdir -p "$RAW_COPY/$date"
  cp "$date_dir"calib_*.txt "$RAW_COPY/$date/"
  for src in "$date_dir"*_sync; do
    [ -d "$src" ] || continue
    dst="$RAW_COPY/$date/$(basename "$src")"
    mkdir -p "$dst"
    cp -r "$src/velodyne_points" "$src/image_02" "$src/oxts" "$dst/"
    cp "$src/tracklet_labels.xml" "$dst/" 2>/dev/null || true
    echo "copied $dst"
  done
done

"$VENV/bin/python" -c "import torch, rclpy, cv_bridge, vision_msgs, multimodal_transformer; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_DONE
