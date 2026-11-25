# Конфігурація

Усі файли лежать у `config/`. Шляхи можна передавати відносно `config/` (`--dataset dataset.yaml`)
або абсолютні. У рядкових значеннях розкриваються змінні оточення (`${KITTI_ROOT}`) та `~`,
тому машинно-залежні шляхи не потрапляють у репозиторій.

## `dataset.yaml` — підготовка та завантаження даних

| Параметр | Тип | За замовч. | Призначення |
| :-- | :-- | :-- | :-- |
| `kitti_root` | шлях | `${KITTI_ROOT}` | корінь KITTI Object (містить `training/`) |
| `crops_dir` | шлях | `${KITTI_ROOT}/crops` | куди `build_dataset` пише `images/`, `lidar/`, `index.csv` |
| `class_names` | список | `[Car, Pedestrian, Cyclist]` | класи моделі; індекс у списку = id мітки |
| `index_file` | рядок | `index.csv` | індекс кропів; `index_with_background.csv` додає клас Background (`config/dataset_background.yaml`) |
| `class_map` | словник | див. файл | тип KITTI → клас проєкту; відсутні типи (Tram, Misc, DontCare) пропускаються |
| `min_points` | int | 5 | об'єкти з меншою кількістю точок в OBB відкидаються |
| `image_padding` | int | 5 | пікселі контексту навколо проєкції бокса |
| `image_size` | int | 224 | розмір входу візуального енкодера |
| `intensity_scale` | float | 1.0 | множник інтенсивності (KITTI 0..1; для 0..255 вкажіть 0.00392) |
| `val_ratio` | float | 0.2 | частка *кадрів* у валідації |
| `split_seed` | int | 42 | зерно розбиття train/val |

## `model.yaml` — архітектура (`model:`)

| Параметр | За замовч. | Призначення |
| :-- | :-- | :-- |
| `embed_dim` | 128 | ширина токенів обох модальностей |
| `nhead` | 8 | кількість голів уваги (`embed_dim % nhead == 0`) |
| `num_layers` | 3 | шари Transformer Encoder |
| `ff_dim` | 256 | прихований розмір feed-forward блоку |
| `dropout` | 0.1 | dropout в енкодері та голові |
| `point_features` | 4 | ознаки точки: x, y, z, інтенсивність |
| `pretrained_backbone` | true | ваги ImageNet для ResNet-18 |
| `freeze_backbone` | true | заморожування ResNet-18 (економія VRAM, менше перенавчання) |
| `modality_dropout` | 0.2 | частка зразків із замаскованим зображенням під час навчання (NFR-05) |

`num_classes` береться з `len(class_names)` автоматично.

## `train.yaml` — навчання

| Параметр | За замовч. | Призначення |
| :-- | :-- | :-- |
| `seed` | 42 | зерно генераторів випадкових чисел |
| `epochs` | 20 | максимум епох |
| `batch_size` | 16 | розмір батчу (поміщається в 4 GB VRAM з AMP) |
| `num_workers` | 4 | процеси DataLoader |
| `num_points` | 1024 | точок на об'єкт |
| `lr` | 3e-4 | піковий learning rate (OneCycle) |
| `weight_decay` | 1e-4 | AdamW |
| `grad_clip` | 1.0 | обмеження норми градієнта |
| `amp` | true | змішана точність на CUDA |
| `label_smoothing` | 0.0 | згладжування міток у CE |
| `early_stopping_patience` | 5 | епох без покращення macro F1 до зупинки |
| `class_weights` | `[1, 5, 10]` | ваги CE: список, `auto` (√ оберненої частоти) або `null` |
| `output_dir` | `runs/default` | `best.pth`, `last.pth`, `training_history.json` |

## `calibration_example.yaml` — калібрування живих сенсорів

| Параметр | Розмір | Призначення |
| :-- | :-- | :-- |
| `P` | 3×4 | матриця проєкції камери |
| `R0` | 3×3 | ректифікація (одинична для моно-камери) |
| `Tr_lidar_to_cam` | 3×4 | жорстке перетворення LiDAR → камера |

## `ros2_pipeline.yaml` — вузли ROS 2

Параметри ROS 2 строго типізовані: числа з плаваючою точкою пишіть з крапкою (`40.0`, не `40`).

### `roi_extraction`

| Параметр | За замовч. | Призначення |
| :-- | :-- | :-- |
| `lidar_topic` / `candidates_topic` | `/lidar/points` / `/roi/candidates` | вхід / вихід |
| `x_range`, `y_range`, `z_range` | `[0,40]`, `[-20,20]`, `[-3,2]` | область інтересу, м, система LiDAR |
| `max_azimuth_deg` | `180` | півкут сектора навколо осі x, град; `50` ≈ поле зору камери KITTI, менше кандидатів |
| `voxel_size` | 0.1 | розмір вокселя для проріджування, м |
| `ground_distance` | 0.2 | поріг інлаєра площини землі (RANSAC), м |
| `ransac_iterations` | 60 | ітерації RANSAC |
| `max_ground_tilt_deg` | 15 | максимальний нахил площини, яку вважають землею |
| `cluster_tolerance` | 0.5 | макс. відстань між сусідніми точками кластера, м |
| `min_cluster_points` / `max_cluster_points` | 10 / 20000 | межі розміру кластера |
| `min_size` / `max_size` | `[0.2,0.2,0.4]` / `[7,3.5,3.5]` | межі розмірів OBB (д/ш/в), м |

### `multimodal_inference`

| Параметр | За замовч. | Призначення |
| :-- | :-- | :-- |
| `lidar_topic`, `image_topic`, `candidates_topic` | див. файл | входи |
| `detections_topic`, `markers_topic` | `/detected_objects`, `.../markers` | виходи |
| `calibration_file` | `config/calibration_example.yaml` | калібрування LiDAR-камера |
| `use_camera` | true | false — тільки LiDAR |
| `max_sync_dt` | 0.05 | допустима різниця stamp LiDAR і камери, с |
| `image_buffer_size` | 10 | кадрів камери в буфері |
| `sync_queue_size` | 10 | черга синхронізатора скан/кандидати; має покривати затримку ROI-екстрактора (10 сканів = 1 с при 10 Гц), інакше скан витісняється до приходу кандидатів і кадр втрачається; затримка не накопичується, бо підписки зберігають лише найновіше повідомлення |
| `image_padding` | 5 | пікселі контексту для патча |
| `min_image_brightness` | 15 | середня яскравість (0..255), нижче — патч вважається відсутнім |
| `class_names` | `[Car, Pedestrian, Cyclist]` | мають збігатися з навчанням |
| `reject_classes` | `[Background]` | класи, які не публікуються (фон, знайдений екстрактором кандидатів) |
| `num_points`, `intensity_scale` | 1024, 1.0 | як у навчанні |
| `min_points` | 5 | менше точок у боксі — кандидат пропускається |
| `min_confidence` | 0.5 | мінімальний score для публікації |
| `backend` | `torch` | `torch`, `onnxruntime` або `tensorrt` |
| `checkpoint` / `onnx` / `engine` | шляхи | модель для відповідного бекенду |
| `device`, `fp16` | `cuda`, false | для бекенду `torch`; повний FP16 на GTX 1650 повільніший за FP32 і може переповнюватися (NaN) |
| `max_batch` | 32 | макс. об'єктів в одному батчі (≤ max профілю TensorRT) |
| `journal_db` | `""` | шлях до SQLite-журналу; порожньо — вимкнено |
| `session_location` | `""` | опис місця сеансу для журналу |

### `kitti_player`

| Параметр | За замовч. | Призначення |
| :-- | :-- | :-- |
| `kitti_split_dir` | `""` | `<kitti_root>/training` (окремі кадри KITTI Object) |
| `kitti_raw_dir` | `""` | поїздка KITTI raw `<date>/<date>_drive_XXXX_sync`; має пріоритет над `kitti_split_dir` |
| `rate_hz` | 10.0 | частота публікації |
| `lidar_frame` | `lidar` | `frame_id` скану |
| `world_frame` | `world` | батьківський кадр TF `world → lidar` (лише поїздки raw) |
| `path_topic` | `/ego/path` | пройдений шлях, `nav_msgs/Path` (лише поїздки raw) |
| `loop` | true | повтор після останнього кадру |
| `calibration_out` | `/tmp/kitti_calibration.yaml` | куди записати калібрування |
