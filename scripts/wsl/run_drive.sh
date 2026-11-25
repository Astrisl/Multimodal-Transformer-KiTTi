#!/usr/bin/env bash
# Step 3 of 3: replays a KITTI raw drive through the ROS 2 pipeline with RViz2, measures
# the output rate of /detected_objects and, with RECORD=1, records RViz2 to an mp4.
#   wsl -d Ubuntu-22.04 -u root -- bash scripts/wsl/run_drive.sh [drive] [seconds]
#   drive   e.g. 2011_09_26_drive_0009_sync (default); looked up in ~/kitti_raw/<date>/
#   RECORD=1      RViz2 runs on a virtual display (Xvfb) and is recorded with ffmpeg
#   OUT=<dir>     logs and video (default runs/ros2_drive/<drive>)
# RViz2 uses software OpenGL: with hardware OpenGL it crashes under WSLg.
set -eo pipefail
source /opt/ros/humble/setup.bash
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-$HOME/mmt-venv}
DRIVE=${1:-2011_09_26_drive_0009_sync}
DURATION=${2:-200}
DRIVE_DIR="${RAW_COPY:-$HOME/kitti_raw}/${DRIVE:0:10}/$DRIVE"
OUT=${OUT:-$REPO/runs/ros2_drive/$DRIVE}
[ -d "$DRIVE_DIR" ] || { echo "drive not found: $DRIVE_DIR (run setup_project.sh)"; exit 1; }
mkdir -p "$OUT"
cd "$REPO"
export PYTHONUNBUFFERED=1 LIBGL_ALWAYS_SOFTWARE=1

if [ "${RECORD:-0}" = 1 ]; then
  Xvfb :99 -screen 0 1280x800x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
  XVFB=$!
  trap 'kill $XVFB 2>/dev/null || true' EXIT
  sleep 2
  export DISPLAY=:99
  unset WAYLAND_DISPLAY
fi

timeout -s INT "$DURATION" ros2 launch src/ros2_nodes/launch/scene_analysis.launch.py \
    kitti_raw_dir:="$DRIVE_DIR" rviz_config:="$REPO/config/rviz/kitti_drive.rviz" \
    use_slam:=false python:="$VENV/bin/python" > "$OUT/launch.log" 2>&1 &
LAUNCH=$!

sleep 60  # model loading and CUDA warm-up
if [ "${RECORD:-0}" = 1 ]; then
  ffmpeg -y -loglevel error -f x11grab -video_size 1280x800 -framerate 10 -i :99 -t 60 \
      -c:v libx264 -pix_fmt yuv420p -preset veryfast "$OUT/rviz.mp4" &
fi
timeout -s INT 90 ros2 topic hz /detected_objects > "$OUT/hz_detected_objects.txt" 2>&1 || true
timeout -s INT 20 ros2 topic hz /lidar/points > "$OUT/hz_lidar.txt" 2>&1 || true
wait $LAUNCH || true
echo "detected_objects: $(grep 'average rate' "$OUT/hz_detected_objects.txt" | tail -n 1)"
grep "FPS in" "$OUT/launch.log" | tail -n 3 || true
echo "RUN_DONE -> $OUT"
