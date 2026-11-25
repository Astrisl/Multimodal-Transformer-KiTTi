#!/usr/bin/env bash
# Step 1 of 3: ROS 2 Humble (desktop: rclpy, RViz2) and the packages the project nodes use,
# plus Xvfb/ffmpeg for recording RViz2. Run as root in WSL2 Ubuntu 22.04:
#   wsl -d Ubuntu-22.04 -u root -- bash scripts/wsl/install_ros.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release locales
locale-gen en_US.UTF-8
add-apt-repository -y universe
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg]" \
    "http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y ros-humble-desktop ros-humble-vision-msgs ros-humble-cv-bridge \
    ros-humble-message-filters ros-humble-tf2-ros python3-pip python3-venv xvfb ffmpeg
grep -q "/opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo ROS_INSTALL_DONE
