# ROS 2 API

Вузли проєкту запускаються як Python-модулі (`python3 -m multimodal_transformer.ros2_nodes.<node>`),
параметри — з `config/ros2_pipeline.yaml`. Використовуються лише стандартні типи повідомлень,
тому власний пакет інтерфейсів не потрібен.

## Граф вузлів

```
/lidar/points ──┬──► roi_extraction ──► /roi/candidates ──┐
                ├──► fast_lio (SLAM) ──► /Laser_map, /Odometry, TF
                └─────────────────────────────────────────┼──► multimodal_inference ──► /detected_objects
/camera/image_raw ────────────────────────────────────────┘                        └──► /detected_objects/markers
```

## Топіки

| Топік | Тип | Публікує | Споживає | Опис |
| :-- | :-- | :-- | :-- | :-- |
| `/lidar/points` | `sensor_msgs/PointCloud2` | драйвер LiDAR / `kitti_player` | `roi_extraction`, `multimodal_inference`, FAST-LIO2 | Скан. Поля `x, y, z` + `intensity` або `reflectivity` (інакше інтенсивність 0). QoS: sensor data (best effort). |
| `/camera/image_raw` | `sensor_msgs/Image` | драйвер камери / `kitti_player` | `multimodal_inference` | `bgr8`, `rgb8`, `mono8`, `mono16` (IR → 3 канали). QoS: sensor data. |
| `/roi/candidates` | `vision_msgs/Detection3DArray` | `roi_extraction` | `multimodal_inference` | OBB кандидатів без класу. `header` = заголовок скану (той самий stamp і frame). |
| `/detected_objects` | `vision_msgs/Detection3DArray` | `multimodal_inference` | RViz2, застосунки | Результати класифікації (FR-06). |
| `/detected_objects/markers` | `visualization_msgs/MarkerArray` | `multimodal_inference` | RViz2 | Ребра боксів (`ns=boxes`) і підписи `"<клас> <score>"` (`ns=labels`), колір за класом. |
| `/Laser_map`, `/cloud_registered`, `/Odometry` | `PointCloud2`, `nav_msgs/Odometry` | FAST-LIO2 | RViz2 | Глобальна PCD-карта та одометрія (FR-05). |

### Поля `Detection3D` у `/detected_objects`

| Поле | Значення |
| :-- | :-- |
| `header` | stamp і frame скану (система LiDAR) |
| `id` | порядковий номер детекції в кадрі |
| `bbox.center.position` | центр OBB, м |
| `bbox.center.orientation` | поворот навколо осі z (yaw) як кватерніон |
| `bbox.size` | `x` = довжина, `y` = ширина, `z` = висота, м |
| `results[0].hypothesis.class_id` | `"Car"`, `"Pedestrian"` або `"Cyclist"` |
| `results[0].hypothesis.score` | softmax-імовірність класу, 0..1 |

Детекції з `score < min_confidence` не публікуються.

## Вузли

### `roi_extraction`
Модуль `multimodal_transformer.ros2_nodes.roi_extraction_node`. Кроп області інтересу →
воксельне проріджування → RANSAC видалення землі → евклідова кластеризація → OBB (PCA) →
розширення бокса до землі. Параметри: розділ `roi_extraction` у [configuration.md](configuration.md).
Кожні 100 сканів логує середній час обробки.

### `multimodal_inference`
Модуль `multimodal_transformer.ros2_nodes.inference_node`. Синхронізує скан і кандидатів за
точним stamp (`message_filters.TimeSynchronizer`), бере найближчий кадр камери з буфера,
формує токени й виконує батч-інференс (бекенд `torch` / `onnxruntime` / `tensorrt`).

Обробка помилок:

| Ситуація | Поведінка |
| :-- | :-- |
| немає кадру камери / `|Δt| > max_sync_dt` | LiDAR-only класифікація, попередження (throttled), лічильник `LiDAR-only` |
| патч темніший за `min_image_brightness` | LiDAR-only для цього об'єкта |
| бокс позаду камери / поза кадром | LiDAR-only для цього об'єкта |
| збій конвертації зображення | LiDAR-only, помилка в лог |
| збій інференсу | кадр пропускається, лічильник `errors`, вузол працює далі |
| диск журналу не встигає | кадр журналу відкидається, лічильник `journal dropped` |

Кожні 5 с логує вхідну частоту, час обробки кадру й максимально досяжний FPS (NFR-01).

### `kitti_player`
Модуль `multimodal_transformer.ros2_nodes.kitti_player_node`. Публікує кадри KITTI як
`/lidar/points` і `/camera/image_raw` з однаковим stamp (за замовчуванням 10 Гц) та записує
калібрування у `calibration_out` для вузла інференсу. Для демонстрації без обладнання.

Джерела: окремі кадри KITTI Object (`kitti_split_dir`) або безперервна поїздка KITTI raw
(`kitti_raw_dir`). Для поїздки рух автомобіля з GPS/IMU (OXTS) публікується як TF
`world → lidar` і як `nav_msgs/Path` у `/ego/path`, тож у RViz2 з
`config/rviz/kitti_drive.rviz` (fixed frame `world`, камера слідує за `lidar`) видно, як
автомобіль їде сценою.

## Launch

`src/ros2_nodes/launch/scene_analysis.launch.py`

| Аргумент | За замовчуванням | Опис |
| :-- | :-- | :-- |
| `params_file` | `config/ros2_pipeline.yaml` | параметри вузлів |
| `kitti_split_dir` | `""` | якщо задано — програвання кадрів KITTI Object замість сенсорів |
| `kitti_raw_dir` | `""` | якщо задано — програвання поїздки KITTI raw (з рухом автомобіля в TF) |
| `use_slam` | `true` | запуск FAST-LIO2 (`fast_lio/launch/mapping.launch.py`) |
| `fast_lio_config` | `mid360.yaml` | файл конфігурації FAST-LIO2 |
| `rviz` | `true` | запуск RViz2 |
| `rviz_config` | `config/rviz/scene_analysis.rviz` | конфігурація RViz2; для поїздки — `config/rviz/kitti_drive.rviz` |
| `python` | `python3` | інтерпретатор зі встановленим проєктом (наприклад, venv) |

Записи KITTI не містять сирого IMU потрібної частоти, тому з `kitti_split_dir` / `kitti_raw_dir`
слід вказувати `use_slam:=false`.
