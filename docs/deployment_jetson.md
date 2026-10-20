# Розгортання на NVIDIA Jetson Orin (Nano / NX)

Покрокова інструкція для JetPack 6.x (Ubuntu 22.04, CUDA 12, TensorRT 10) та ROS 2 Humble.
Для JetPack 5 (Ubuntu 20.04) ROS 2 Humble збирається з вихідних кодів або запускається в
контейнері `dustynv/ros:humble-*`.

## 1. Система

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
# максимальна продуктивність (номер режиму залежить від модуля: `nvpmodel -q`)
sudo nvpmodel -m 0 && sudo jetson_clocks
```

## 2. ROS 2 Humble та залежності

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install -y \
  ros-humble-ros-base ros-humble-rviz2 ros-humble-vision-msgs ros-humble-cv-bridge \
  ros-humble-sensor-msgs-py ros-humble-message-filters ros-humble-pcl-ros \
  python3-colcon-common-extensions python3-rosdep
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

## 3. ROS 2 workspace: FAST-LIO2 і драйвер сенсора

Сторонні пакети збираються в окремому workspace (проєкт сам по собі збирати colcon не треба).

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
# драйвер LiDAR (приклад для Livox MID-360; для Velodyne: ros-humble-velodyne)
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
# FAST-LIO2, гілка ROS 2
git clone --recursive -b ROS2 https://github.com/hku-mars/FAST_LIO.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

Конфігурацію сенсора (топіки LiDAR/IMU, extrinsics) задайте у `FAST_LIO/config/<sensor>.yaml`
і передайте її ім'я в launch-аргумент `fast_lio_config`.

## 4. Проєкт

```bash
git clone <repo-url> ~/multimodal-lidar-ir-transformer && cd ~/multimodal-lidar-ir-transformer
python3 -m venv --system-site-packages .venv    # system-site-packages: rclpy, cv_bridge, tensorrt
source .venv/bin/activate
```

PyTorch для Jetson встановлюється з колес NVIDIA, а не з PyPI (версія має відповідати JetPack;
актуальні посилання — у розділі *PyTorch for Jetson* на сайті NVIDIA):

```bash
pip install <torch-*.whl від NVIDIA для вашого JetPack> <відповідний torchvision>
pip install numpy scipy pillow pyyaml tqdm matplotlib onnx
pip install -e . --no-deps
python3 -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
```

`opencv-python-headless` не ставте поверх системного OpenCV з JetPack (конфлікт з `cv_bridge`).

## 5. Модель → TensorRT

Навчання виконується на ПК з GPU; на Jetson копіюється лише `best.pth`.

```bash
python -m multimodal_transformer.model.export --checkpoint best.pth --onnx model.onnx
python -m multimodal_transformer.model.export --onnx model.onnx --engine model.engine --fp16 --max-batch 32
```

Engine прив'язаний до конкретного GPU та версії TensorRT — будуйте його саме на цільовому пристрої.

У `config/ros2_pipeline.yaml` для вузла `multimodal_inference`:

```yaml
    backend: tensorrt
    engine: /home/<user>/multimodal-lidar-ir-transformer/model.engine
    calibration_file: /home/<user>/calibration.yaml   # результат калібрування LiDAR-камера
    intensity_scale: 0.00392                         # для Livox (reflectivity 0..255)
```

## 6. Запуск

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
# драйвер сенсорів (приклад)
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
# пайплайн: FAST-LIO2 + ROI + інференс + RViz2
ros2 launch src/ros2_nodes/launch/scene_analysis.launch.py \
    python:=$PWD/.venv/bin/python fast_lio_config:=mid360.yaml rviz:=false
```

Якщо драйвер публікує інші топіки, змініть `lidar_topic` / `image_topic` у
`config/ros2_pipeline.yaml`. RViz2 зручніше запускати на ПК в тій самій мережі (`ROS_DOMAIN_ID`).

## 7. Перевірка

```bash
ros2 topic hz /detected_objects          # очікується ≥ 20 Гц при 20 Гц сенсорі (NFR-01)
ros2 topic echo /detected_objects --once
```

Вузол інференсу кожні 5 с логує час обробки кадру та досяжний FPS; лічильник `LiDAR-only`
показує, як часто камера була недоступна. Для замірів без обладнання можна запустити пайплайн
на KITTI (`kitti_split_dir:=...`, `use_slam:=false`).

## 8. Zero-copy (опційно)

Для передачі великих хмар точок між вузлами одного хоста без копіювання ввімкніть shared
memory у Cyclone DDS / iceoryx або Fast DDS:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=<профіль з data-sharing/SHM transport>
```
