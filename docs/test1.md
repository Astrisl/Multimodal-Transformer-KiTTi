# Перехід від курсової до магістерської роботи

## 1. Основна відмінність

Курсова робота базується на:

```text
KITTI
  ↓
RGB image
  ↓
Linear projection
  ↓
LiDAR points
  ↓
Linear projection
  ↓
Transformer Encoder
  ↓
Classification
```

Магістерська робота повинна бути побудована навколо більш повної мультимодальної системи:

```text
                    ┌── ResNet-18 ── Image Tokens ──┐
RGB image ──────────┤                                │
                    └────────────────────────────────┤
                                                     ↓
LiDAR ── Point Encoder ── LiDAR Tokens ───────► Transformer
                                                     ↓
                                                 CLS Token
                                                     ↓
                                                MLP Head
                                                     ↓
                                    Car / Pedestrian / Cyclist
```

Додатково система повинна розглядатися не тільки як нейромережева модель, а як програмно-апаратний pipeline:

```text
KITTI
  ↓
Data preprocessing
  ↓
Model training
  ↓
ONNX
  ↓
TensorRT
  ↓
ROS 2
  ↓
Real-time inference
  ↓
Jetson Orin
  ↓
RViz2 / Detection3D
```

---


# 2. Що залишити з курсовій

Наступні частини курсової є актуальною основою магістерської:

- датасет KITTI;
- RGB + LiDAR;
- калібрування сенсорів;
- системи координат;
- проєкція LiDAR → Camera;
- Oriented Bounding Box;
- LiDAR cropping;
- нормалізація точкових хмар;
- ресемплінг до 1024 точок;
- class imbalance;
- Weighted Cross Entropy;
- Transformer;
- Intermediate Fusion;
- Precision;
- Recall;
- F1-Score;
- експериментальна оцінка;
- аналіз помилок.

---

# 3. Що потрібно змінити або прибрати

| Частина курсової | Що робити в магістерській |
|---|---|
| RGB + LiDAR | Залишити |
| KITTI | Залишити |
| Calibration | Залишити |
| LiDAR cropping | Залишити |
| 1024 points | Залишити |
| Weighted Cross Entropy | Залишити та дослідити |
| Transformer | Залишити |
| Intermediate Fusion | Залишити |
| Простий Linear image projection | Замінити на ResNet-18 |
| Один image token | Замінити на 49 image tokens |
| Простий LiDAR projection | Замінити на Point Encoder |
| Accuracy як головна метрика | Зробити Macro F1 основною |
| Один приклад класифікації Car | Прибрати як основний доказ |
| 95.57% Accuracy | Залишити як baseline/version 1 |
| 25–30 ms на GTX 1650 | Залишити як baseline |
| IR | Прибрати|
| ROS 2 | Додати |
| ONNX | Додати |
| TensorRT | Додати |
| Jetson Orin | Додати |
| FAST-LIO2 | Додати |
| RViz2 | Додати |
| Modality Dropout | Додати |
| LiDAR-only режим | Додати |
| RGB-only baseline | Додати |
| LiDAR-only baseline | Додати |
| Ablation Study | Обов'язково додати |
| Robustness Study | Додати |
| FPS / latency / memory | Додати |

---

# 4. Нова структура магістерської

# ВСТУП

У вступі необхідно описати:

- актуальність мультимодального сприйняття;
- проблеми використання тільки RGB;
- проблеми використання тільки LiDAR;
- переваги fusion;
- актуальність Transformer;
- необхідність real-time inference;
- мету роботи;
- завдання;
- об'єкт дослідження;
- предмет дослідження;
- методи дослідження;
- наукову новизну;
- практичне значення.

---

# РОЗДІЛ 1. АНАЛІЗ МЕТОДІВ МУЛЬТИМОДАЛЬНОГО СПРИЙНЯТТЯ

## 1.1. Системи сприйняття автономного транспорту

Описати:

- камери;
- LiDAR;
- особливості RGB;
- особливості LiDAR;
- переваги та недоліки кожного сенсора;
- необхідність комбінування різних модальностей.

Не потрібно робити великий окремий блок про Radar, якщо він не використовується у власній системі.

---

## 1.2. Методи обробки 3D-хмар точок

Розглянути:

- Point Cloud;
- PointNet;
- PointNet++;
- voxel-based підходи;
- BEV;
- Point Transformer;
- Transformer-based approaches.

Головна мета цього підрозділу — обґрунтувати вибір способу обробки LiDAR.

---

## 1.3. Методи мультимодального злиття даних

Описати:

### Early Fusion

Об'єднання даних на ранньому етапі.

Переваги:

- простота;
- безпосереднє об'єднання даних.

Недоліки:

- висока залежність від calibration;
- проблеми з розрідженістю LiDAR.

### Intermediate Fusion

Окреме формування ознак для кожної модальності з подальшим fusion.

Це основний підхід власної системи.

### Late Fusion

Окрема обробка кожної модальності з об'єднанням фінальних рішень.

---

## 1.4. Transformer-архітектури для мультимодального сприйняття

Описати:

- Self-Attention;
- Multi-Head Attention;
- Transformer Encoder;
- token representation;
- CLS token;
- modality embeddings;
- cross-modal interaction.

Особливу увагу приділити тому, чому Transformer використовується саме для fusion.

---

## 1.5. Постановка задачі та обґрунтування вибору архітектури

Сформулювати:

> Розробити та дослідити мультимодальну систему розпізнавання об'єктів дорожньої сцени, яка використовує RGB-зображення та 3D-хмари точок LiDAR, формує їхні ознакові представлення у вигляді токенів та виконує їхнє спільне опрацювання за допомогою Transformer Encoder.

---

# РОЗДІЛ 2. ПІДГОТОВКА ТА ГЕОМЕТРИЧНЕ УЗГОДЖЕННЯ ДАНИХ

## 2.1. Датасет KITTI та структура даних

Описати:

- image_2;
- Velodyne;
- calibration;
- labels;
- кількість даних;
- формат RGB;
- формат LiDAR;
- структуру файлів.

---

## 2.2. Системи координат і калібрування

Описати:

- координатну систему LiDAR;
- координатну систему камери;
- calibration matrices;
- `P2`;
- `R0_rect`;
- `Tr_velo_to_cam`.

---

## 2.3. Геометричне перетворення LiDAR → Camera

Показати математичне перетворення:

```text
LiDAR coordinates
      ↓
Rectified camera coordinates
      ↓
Image coordinates
```

Пояснити, як визначається відповідність між 3D-точками та областю зображення.

---

## 2.4. Виділення об'єктів за Oriented Bounding Box

Описати:

- 3D bounding boxes;
- параметри `h, w, l, ry`;
- визначення точок, які належать об'єкту;
- LiDAR cropping.

---

## 2.5. Формування мультимодальних ROI

Для кожного об'єкта формується:

```text
RGB object crop
+
LiDAR object points
```

Ці дані стають спільним входом мультимодальної моделі.

---

## 2.6. Нормалізація та ресемплінг хмар точок

Використовується фіксований розмір:

```text
1024 × 4
```

де:

```text
[x, y, z, intensity]
```

Якщо точок більше 1024:

```text
random sampling
```

Якщо менше:

```text
cyclic padding
```

---

# РОЗДІЛ 3. РОЗРОБКА МУЛЬТИМОДАЛЬНОЇ TRANSFORMER-СИСТЕМИ

## 3.1. Загальна архітектура системи

Основна схема:

```text
                    ┌── ResNet-18 ── 49 Image Tokens ──┐
                    │                                    │
RGB image ──────────┤                                    │
                    │                                    ↓
                    │                              Modality Embedding
                    │                                    │
                    │                                    ↓
LiDAR ── Point Encoder ── 1024 LiDAR Tokens ──────► Transformer
                                                         ↓
                                                     CLS Token
                                                         ↓
                                                      MLP Head
                                                         ↓
                                             Classification Result
```

Загальна кількість токенів:

```text
1 CLS
+
49 image tokens
+
1024 LiDAR tokens
=
1074 tokens
```

---

## 3.2. Візуальний енкодер ResNet-18

Описати:

```text
224 × 224 × 3
       ↓
ResNet-18
       ↓
7 × 7 × 512
       ↓
49 spatial tokens
       ↓
Linear projection
       ↓
49 × 128
```

На відміну від курсової, зображення більше не перетворюється в один великий flattened vector.

---

## 3.3. LiDAR Point Encoder

Вхід:

```text
1024 × 4
```

де кожна точка:

```text
x
y
z
intensity
```

Приклад:

```text
1024 × 4
   ↓
MLP 4 → 64
   ↓
MLP 64 → 128
   ↓
1024 × 128
```

Таким чином кожна точка стає окремим LiDAR token.

---

## 3.4. Формування токенів та modality embeddings

Формується послідовність:

```text
[CLS]
[IMG1]
[IMG2]
...
[IMG49]
[POINT1]
...
[POINT1024]
```

До токенів додається інформація про модальність:

```text
Image modality
LiDAR modality
```

Це дозволяє Transformer розрізняти походження токенів.

---

## 3.5. Transformer Encoder та мультимодальне fusion

Параметри:

```text
d_model = 128
layers = 3
attention heads = 8
```

Transformer виконує спільну обробку:

```text
Image tokens
      +
LiDAR tokens
      ↓
Self-Attention
      ↓
Cross-modal interaction
      ↓
CLS representation
```

CLS token використовується для фінальної класифікації.

---

## 3.6. Modality Dropout та LiDAR-only режим

Під час навчання частина зразків обробляється без image tokens.

Наприклад:

```text
20% samples
      ↓
Image tokens masked
      ↓
LiDAR-only inference
```

Мета:

- зменшити залежність від однієї модальності;
- підвищити robustness;
- дослідити поведінку системи при втраті камери.

---

## 3.7. Навчання та оптимізація моделі

Описати:

- train/validation split;
- batch size;
- optimizer;
- learning rate;
- number of epochs;
- scheduler;
- Weighted Cross Entropy;
- class weights;
- checkpoint selection;
- augmentation.

Для оцінювання основною метрикою бажано використовувати:

```text
Macro F1
```

Accuracy використовувати як додаткову метрику.

---

## 3.8. Експорт моделі в ONNX та TensorRT

Pipeline:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ↓
FP16
   ↓
Edge inference
```

Дослідити:

- latency;
- FPS;
- memory;
- inference time;
- відмінність PyTorch / ONNX / TensorRT.

---

## 3.9. Real-time архітектура ROS 2

Описати:

```text
Camera
   ↓
ROS 2
   ↓
Image preprocessing
   ↓
       ┌───────────────┐
LiDAR ─┤               │
       │ Multimodal    │
Camera ┤ Transformer   │
       │               │
       └───────┬───────┘
               ↓
        Detection3DArray
               ↓
             RViz2
```

Також описати:

- ROS 2 nodes;
- topics;
- messages;
- synchronization;
- inference node;
- visualization;
- logging.

---

# РОЗДІЛ 4. ЕКСПЕРИМЕНТАЛЬНЕ ДОСЛІДЖЕННЯ

## 4.1. Методика проведення експериментів

Вказати:

- hardware;
- GPU;
- CPU;
- RAM;
- software;
- Python;
- PyTorch;
- CUDA;
- dataset split;
- training parameters;
- random seed.

---

## 4.2. Результати базової мультимодальної моделі

Показати:

- Accuracy;
- Precision;
- Recall;
- F1;
- Macro F1;
- confusion matrix;
- training loss;
- validation loss.

Старий результат:

```text
Validation Accuracy ≈ 95.57%
```

можна залишити як результат попередньої/version 1 моделі, але не подавати як головний результат нової магістерської моделі без повторного експерименту.

---

# 4.3. Порівняння RGB-only, LiDAR-only та Fusion

Обов'язково провести:

```text
Experiment A:
RGB only

Experiment B:
LiDAR only

Experiment C:
RGB + LiDAR

Experiment D:
RGB + LiDAR + Transformer Fusion
```

Порівняти:

| Model | RGB | LiDAR | Transformer | Macro F1 |
|---|---:|---:|---:|---:|
| RGB-only | ✓ | | | |
| LiDAR-only | | ✓ | | |
| Simple Fusion | ✓ | ✓ | | |
| Transformer Fusion | ✓ | ✓ | ✓ | |

Це дозволить довести користь запропонованої архітектури.

---

# 4.4. Ablation Study

Дослідити вплив окремих компонентів:

- без modality embeddings;
- без CLS token;
- без modality dropout;
- без Transformer;
- інша кількість Transformer layers;
- інша кількість attention heads;
- різна кількість LiDAR tokens.

Приклад:

| Configuration | Macro F1 |
|---|---:|
| Full model | |
| Without modality embedding | |
| Without modality dropout | |
| Without Transformer | |
| LiDAR only | |
| RGB only | |

---

# 4.5. Дослідження стійкості до втрати модальності

Перевірити:

```text
Normal:
RGB + LiDAR

Missing RGB:
LiDAR only

Sparse LiDAR:
1024 points
512 points
256 points
128 points
64 points
```

Побудувати графік:

```text
Number of LiDAR points
        ↓
Macro F1
```

Це дозволить оцінити robustness моделі.

---

# 4.6. Аналіз продуктивності та real-time inference

Порівняти:

| Backend | Latency | FPS | Memory |
|---|---:|---:|---:|
| PyTorch | | | |
| ONNX | | | |
| TensorRT FP16 | | | |

Якщо експерименти проводяться на Jetson Orin:

окремо показати:

- inference latency;
- preprocessing latency;
- total pipeline latency;
- FPS;
- GPU memory;
- CPU utilization;
- GPU utilization.

---

# 4.7. Аналіз помилок

Дослідити помилки залежно від:

- distance;
- object size;
- LiDAR density;
- occlusion;
- truncation;
- overlap;
- visibility;
- якості RGB.

Особливо проаналізувати:

```text
Car
Pedestrian
Cyclist
```

---

# 4.8. Обмеження системи та напрями подальших досліджень

Описати:

- обмежену кількість класів;
- залежність від calibration;
- залежність від KITTI;
- domain gap;
- computational cost;
- велику кількість Transformer tokens;
- обмежену кількість LiDAR points;
- відсутність складних погодних умов;
- необхідність тестування на реальних сенсорах.

---

# 5. Наукова новизна

Наукову новизну варто будувати навколо реально виконаних експериментів.

Можливі положення:

1. Розроблено мультимодальну архітектуру, яка представляє RGB та LiDAR як спільну послідовність токенів і виконує їхнє спільне опрацювання за допомогою Transformer Encoder.

2. Використано modality embeddings для явного розрізнення візуальних та геометричних ознак під час мультимодального fusion.

3. Досліджено вплив втрати однієї з модальностей із використанням modality dropout та LiDAR-only режиму.

4. Досліджено залежність якості класифікації від щільності LiDAR-точок.

5. Досліджено можливість оптимізації мультимодальної моделі для real-time inference із використанням ONNX/TensorRT.

Не потрібно заявляти ці положення як наукову новизну, якщо відповідні експерименти фактично не проведені.

---

# 7. Практичне значення

Рекомендується сформулювати практичне значення через створену систему:

- програмний pipeline підготовки KITTI;
- multimodal Transformer model;
- RGB + LiDAR fusion;
- ROS 2 integration;
- ONNX export;
- TensorRT optimization;
- можливість deployment на edge-платформі;
- visualization через RViz2;
- logging результатів.

---

# 8. Головна відмінність магістерської

## Магістерська

```text
Проаналізувати existing approaches
              ↓
Запропонувати architecture
              ↓
RGB Encoder + LiDAR Encoder
              ↓
Transformer Fusion
              ↓
Навчити модель
              ↓
RGB-only
LiDAR-only
Fusion
              ↓
Ablation Study
              ↓
Robustness Study
              ↓
Sparse LiDAR Study
              ↓
ONNX / TensorRT
              ↓
Real-time / ROS 2
              ↓
Порівняння результатів
              ↓
Аналіз обмежень
```

Саме ця структура робить роботу **магістерським дослідженням**, а не розширеною курсовою.

---

# 10. Пріоритет змін

## 🔴 Обов'язково

1. Прибрати/замінити IR, якщо його реально немає.
2. Описати новий ResNet-18 image encoder.
3. Описати Point Encoder.
4. Описати token-based fusion.
5. Описати CLS token.
6. Описати modality embeddings.
7. Описати modality dropout.
8. Додати RGB-only baseline.
9. Додати LiDAR-only baseline.
10. Додати Fusion baseline.
11. Провести Ablation Study.
12. Перейти на Macro F1 як основну метрику.
13. Провести аналіз robustness.

## 🟠 Дуже бажано

14. ONNX.
15. TensorRT.
16. FP16.
17. ROS 2.
18. Jetson Orin.
19. FPS / latency / memory.
20. Sparse LiDAR experiment.
21. Confusion matrix.
22. Error analysis.

## 🟢 Додатково

23. FAST-LIO2.
24. RViz2.
25. SQLite logging.
26. Реальні сенсори.
27. Порівняння з сучасними published approaches.
