"""Figures of the master's thesis (Ukrainian labels) from the project's data and results.

Run from the repository root after training, `run.ps1 eval` and `run.ps1 experiments`:

    python thesis/make_figures.py

Writes PNG files (200 dpi) to thesis/figures/.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "default"
EXP = RUN / "experiments"
CROPS = Path(os.environ.get("KITTI_ROOT", ROOT / "data" / "kitti")) / "crops"
OUT = ROOT / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)
CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLASS_UA = {
    "Car": "Car (автомобіль)",
    "Pedestrian": "Pedestrian (пішохід)",
    "Cyclist": "Cyclist (велосипедист)",
}
C_FUSED, C_LIDAR = "#1f6fb4", "#e07b1a"
C_CLASS = {"Car": "#1f6fb4", "Pedestrian": "#2f9e44", "Cyclist": "#d9480f"}


def read_csv(path: Path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- diagrams


def box(ax, x, y, w, h, text, color="#e8f1fb", edge="#1f4e79", size=11, bold=False):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.2,
    )  # fmt: skip
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", wrap=True)  # fmt: skip


def arrow(ax, p, q, text=None, style="-|>", color="#333333", rad=0.0, size=10):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, linewidth=1.2,
                        connectionstyle=f"arc3,rad={rad}")  # fmt: skip
    )
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", va="bottom",
                fontsize=size, style="italic")  # fmt: skip


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def fig_fusion_strategies():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    titles = ["а) раннє злиття", "б) проміжне злиття", "в) пізнє злиття"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title)
    img, lid, fus, net = "#fff3d6", "#e3f4e3", "#f3e5f5", "#e8f1fb"
    a = axes[0]
    box(a, 1, 5.3, 1.6, 0.7, "RGB", img)
    box(a, 3, 5.3, 1.6, 0.7, "LiDAR", lid)
    box(a, 2, 3.9, 3.2, 0.8, "Об'єднання сирих даних\n(RGB-D, painting)", fus)
    box(a, 2, 2.3, 2.6, 0.8, "Єдина мережа", net)
    box(a, 2, 0.8, 2.2, 0.7, "Клас", "#ffffff")
    for p, q in [
        ((1, 4.95), (1.6, 4.3)),
        ((3, 4.95), (2.4, 4.3)),
        ((2, 3.5), (2, 2.7)),
        ((2, 1.9), (2, 1.15)),
    ]:
        arrow(a, p, q)
    b = axes[1]
    box(b, 1, 5.3, 1.6, 0.7, "RGB", img)
    box(b, 3, 5.3, 1.6, 0.7, "LiDAR", lid)
    box(b, 1, 3.9, 1.7, 0.8, "Енкодер\nзображення", net)
    box(b, 3, 3.9, 1.7, 0.8, "Енкодер\nточок", net)
    box(b, 2, 2.3, 3.4, 0.8, "Злиття ознак\n(Transformer, конкатенація)", fus)
    box(b, 2, 0.8, 2.2, 0.7, "Клас", "#ffffff")
    for p, q in [
        ((1, 4.95), (1, 4.3)),
        ((3, 4.95), (3, 4.3)),
        ((1, 3.5), (1.6, 2.7)),
        ((3, 3.5), (2.4, 2.7)),
        ((2, 1.9), (2, 1.15)),
    ]:
        arrow(b, p, q)
    c = axes[2]
    box(c, 1, 5.3, 1.6, 0.7, "RGB", img)
    box(c, 3, 5.3, 1.6, 0.7, "LiDAR", lid)
    box(c, 1, 3.9, 1.7, 0.8, "Мережа 1", net)
    box(c, 3, 3.9, 1.7, 0.8, "Мережа 2", net)
    box(c, 1, 2.5, 1.7, 0.6, "Прогноз 1", "#ffffff")
    box(c, 3, 2.5, 1.7, 0.6, "Прогноз 2", "#ffffff")
    box(c, 2, 1.1, 3.2, 0.8, "Об'єднання рішень\n(голосування, NMS)", fus)
    for p, q in [
        ((1, 4.95), (1, 4.3)),
        ((3, 4.95), (3, 4.3)),
        ((1, 3.5), (1, 2.8)),
        ((3, 3.5), (3, 2.8)),
        ((1, 2.2), (1.6, 1.5)),
        ((3, 2.2), (2.4, 1.5)),
    ]:
        arrow(c, p, q)
    save(fig, "fig1_fusion_strategies.png")


def fig_encoder_layer():
    fig, ax = canvas(5.2, 6.4, (0, 5), (0, 9.4))
    box(ax, 2.5, 0.6, 3.4, 0.7, "Вхідні токени  X  [B, T, d]", "#ffffff")
    box(ax, 2.5, 2.0, 2.6, 0.7, "LayerNorm", "#fff3d6")
    box(ax, 2.5, 3.3, 3.4, 0.8, "Multi-Head Self-Attention\n(h = 8, маска ключів)", "#e8f1fb")
    box(ax, 2.5, 4.6, 0.6, 0.6, "+", "#f3e5f5", size=14)
    box(ax, 2.5, 5.8, 2.6, 0.7, "LayerNorm", "#fff3d6")
    box(ax, 2.5, 7.0, 3.4, 0.8, "FFN: Linear → GELU → Linear\n(128 → 256 → 128)", "#e3f4e3")
    box(ax, 2.5, 8.3, 0.6, 0.6, "+", "#f3e5f5", size=14)
    for p, q in [((2.5, 0.95), (2.5, 1.65)), ((2.5, 2.35), (2.5, 2.9)), ((2.5, 3.7), (2.5, 4.3)),
                 ((2.5, 4.9), (2.5, 5.45)), ((2.5, 6.15), (2.5, 6.6)), ((2.5, 7.4), (2.5, 8.0)),
                 ((2.5, 8.6), (2.5, 9.3))]:  # fmt: skip
        arrow(ax, p, q)
    ax.plot([2.5, 4.5, 4.5, 2.8], [1.3, 1.3, 4.6, 4.6], color="#777777", lw=1.1)
    ax.plot([2.5, 4.5, 4.5, 2.8], [5.15, 5.15, 8.3, 8.3], color="#777777", lw=1.1)
    ax.text(4.6, 2.9, "залишкове\nз'єднання", fontsize=10, style="italic", va="center")
    ax.text(0.2, 9.0, "× 3 шари", fontsize=12, fontweight="bold")
    save(fig, "fig1_encoder_layer.png")


def fig_coordinate_chain():
    fig, ax = canvas(12, 2.6, (0, 12), (0.6, 2.6))
    box(
        ax,
        1.6,
        1.6,
        2.8,
        1.5,
        "Система LiDAR\nx — вперед, y — вліво,\nz — вгору\n(x, y, z, 1)",
        "#e3f4e3",
        size=12,
    )
    box(
        ax,
        6.0,
        1.6,
        2.9,
        1.5,
        "Ректифікована\nсистема камери\nx — вправо, y — вниз,\nz — вперед",
        "#e8f1fb",
        size=12,
    )
    box(ax, 10.4, 1.6, 2.8, 1.5, "Площина\nзображення\n(u, v), пікселі", "#fff3d6", size=12)
    arrow(ax, (3.05, 1.6), (4.5, 1.6))
    arrow(ax, (7.5, 1.6), (8.95, 1.6))
    ax.text(3.78, 1.85, "R0_rect · Tr_velo_to_cam", ha="center", fontsize=12, style="italic")
    ax.text(8.22, 1.85, "P2, ділення\nна глибину", ha="center", fontsize=12, style="italic")
    save(fig, "fig2_coordinate_chain.png")


def fig_system_architecture():
    fig, ax = canvas(13, 6.4, (0, 13), (0, 6.75))
    ax.add_patch(FancyBboxPatch((0.15, 4.15), 12.7, 2.45, boxstyle="round,pad=0.02",
                                facecolor="#fafafa", edgecolor="#999999", ls="--"))  # fmt: skip
    ax.add_patch(FancyBboxPatch((0.15, 0.1), 12.7, 3.85, boxstyle="round,pad=0.02",
                                facecolor="#fafafa", edgecolor="#999999", ls="--"))  # fmt: skip
    ax.text(0.35, 6.3, "Офлайн-контур навчання (ПК з GPU)", fontsize=12, fontweight="bold")
    ax.text(
        0.35, 3.65, "Контур реального часу (ROS 2, Jetson Orin)", fontsize=12, fontweight="bold"
    )
    y = 5.05
    box(ax, 1.3, y, 1.9, 1.2, "KITTI\nimage_2, velodyne,\ncalib, label_2", "#fff3d6", size=9)
    box(ax, 3.7, y, 2.0, 1.2, "build_dataset\nOBB-кропінг,\n2D-патч", "#e8f1fb", size=9)
    box(ax, 6.1, y, 1.9, 1.2, "crops/\nimages, lidar,\nindex.csv", "#fff3d6", size=9)
    box(ax, 8.5, y, 2.0, 1.2, "Dataset + Trainer\nзважена CE,\nmacro F1", "#e8f1fb", size=9)
    box(ax, 11.4, y, 2.4, 1.2, "best.pth → ONNX\n→ TensorRT FP16", "#f3e5f5", size=9)
    for a, b in [(2.25, 2.7), (4.7, 5.15), (7.05, 7.5), (9.5, 10.2)]:
        arrow(ax, (a, y), (b, y))
    box(ax, 1.3, 2.85, 1.9, 0.8, "LiDAR", "#e3f4e3")
    box(ax, 1.3, 1.65, 1.9, 0.8, "Камера", "#fff3d6")
    box(ax, 4.3, 2.85, 2.4, 0.9, "roi_extraction\nRANSAC, кластери, OBB", "#e8f1fb", size=9)
    box(ax, 4.3, 0.65, 2.4, 0.7, "FAST-LIO2 (SLAM)", "#e8f1fb", size=9)
    box(ax, 8.0, 2.2, 2.8, 1.5, "multimodal_inference\nсинхронізація, токени,\nTensorRT / ONNX / torch",
        "#f3e5f5", size=9)  # fmt: skip
    box(ax, 11.6, 2.85, 2.0, 0.8, "SQLite-журнал", "#ffffff", size=10)
    box(ax, 11.6, 0.65, 2.0, 0.8, "RViz2", "#ffffff")
    arrow(ax, (2.25, 2.85), (3.1, 2.85))
    arrow(ax, (2.25, 2.6), (3.2, 0.95))
    arrow(ax, (2.25, 3.05), (6.6, 2.75), rad=-0.3)
    arrow(ax, (5.5, 2.75), (6.6, 2.45), "/roi/candidates", size=9)
    arrow(ax, (2.25, 1.65), (6.6, 1.75))
    arrow(ax, (9.4, 2.6), (10.6, 2.85))
    arrow(ax, (9.4, 1.6), (10.6, 0.8), "/detected_objects", size=9)
    arrow(ax, (5.5, 0.6), (10.6, 0.6), "/Laser_map", size=9)
    arrow(ax, (11.4, 4.45), (8.7, 2.95), "engine", color="#7b3fa0", rad=-0.15, size=9)
    save(fig, "fig3_system_architecture.png")


def fig_model_architecture():
    fig, ax = canvas(13, 5.4, (0, 13), (1.2, 6.9))
    box(ax, 1.2, 5.6, 2.0, 0.9, "RGB-патч\n3×224×224", "#fff3d6", size=10)
    box(ax, 1.2, 2.0, 2.0, 0.9, "Точки LiDAR\n1024×4", "#e3f4e3", size=10)
    box(ax, 3.9, 5.6, 2.6, 0.9, "ResNet-18 (заморожений)\n512×7×7", "#e8f1fb", size=10)
    box(ax, 3.9, 2.0, 2.6, 0.9, "PointEncoder\nMLP 4→64→128", "#e8f1fb", size=10)
    box(ax, 6.7, 5.6, 2.3, 0.9, "Linear 512→128\n49 токенів", "#e8f1fb", size=10)
    box(ax, 6.7, 2.0, 2.3, 0.9, "1024 токени\n×128", "#e8f1fb", size=10)
    box(ax, 6.7, 3.8, 2.3, 0.7, "CLS-токен", "#f3e5f5", size=10)
    box(ax, 9.0, 4.7, 1.2, 0.6, "+ e_img", "#fff3d6", size=10)
    box(ax, 9.0, 2.9, 1.2, 0.6, "+ e_lidar", "#e3f4e3", size=10)
    box(ax, 10.75, 3.8, 1.5, 3.6, "Transformer\nEncoder\n3 шари,\n8 голів,\nd = 128\n\n1074\nтокени",
        "#f3e5f5", size=10)  # fmt: skip
    box(ax, 12.35, 3.8, 1.0, 2.0, "MLP\nголова\n→\n3 класи", "#ffffff", size=10)
    arrow(ax, (2.2, 5.6), (2.6, 5.6))
    arrow(ax, (2.2, 2.0), (2.6, 2.0))
    arrow(ax, (5.2, 5.6), (5.55, 5.6))
    arrow(ax, (5.2, 2.0), (5.55, 2.0))
    arrow(ax, (7.85, 5.4), (8.4, 4.9))
    arrow(ax, (7.85, 2.2), (8.4, 2.7))
    arrow(ax, (9.6, 4.7), (10.0, 4.4))
    arrow(ax, (9.6, 2.9), (10.0, 3.2))
    arrow(ax, (7.85, 3.8), (10.0, 3.8))
    arrow(ax, (11.5, 3.8), (11.85, 3.8))
    ax.text(
        10.75,
        6.3,
        "маска image_valid\n(відсутня / темна камера)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    arrow(ax, (10.75, 6.0), (10.75, 5.65), color="#999999")
    save(fig, "fig3_model_architecture.png")


def fig_ros_graph():
    fig, ax = canvas(12.5, 5.0, (0, 12.5), (0.3, 5.4))
    box(ax, 1.3, 3.2, 2.2, 0.8, "Драйвер LiDAR /\nkitti_player", "#e3f4e3", size=10)
    box(ax, 1.3, 1.0, 2.2, 0.8, "Драйвер камери /\nkitti_player", "#fff3d6", size=10)
    box(ax, 5.0, 4.8, 2.4, 0.8, "fast_lio (SLAM)", "#e8f1fb", size=11)
    box(ax, 5.0, 3.4, 2.4, 0.8, "roi_extraction", "#e8f1fb", size=11)
    box(ax, 8.6, 2.0, 2.6, 1.0, "multimodal_inference", "#f3e5f5", size=11)
    box(ax, 11.5, 4.1, 1.7, 0.8, "RViz2", "#ffffff", size=11)
    box(ax, 11.5, 1.0, 1.7, 0.8, "SQLite", "#ffffff", size=11)
    ax.text(2.5, 3.75, "/lidar/points", fontsize=9, style="italic")
    arrow(ax, (2.4, 3.4), (3.8, 3.4))
    arrow(ax, (2.4, 3.55), (3.8, 4.7))
    arrow(ax, (2.4, 2.95), (7.3, 2.3))
    arrow(ax, (6.2, 3.2), (7.7, 2.5), "/roi/candidates", size=9)
    arrow(ax, (2.4, 1.0), (7.3, 1.75), "/camera/image_raw", size=9)
    arrow(ax, (6.2, 4.8), (10.65, 4.2), "/Laser_map, /Odometry", size=9)
    arrow(ax, (9.9, 2.4), (10.65, 3.8), "/detected_objects\n+ /markers", size=8)
    arrow(ax, (9.9, 1.7), (10.65, 1.1))
    save(fig, "fig3_ros_graph.png")


def fig_module_structure():
    fig, ax = canvas(12.5, 5.6, (0, 12.5), (0, 6))
    mods = [
        (2.0, 4.8, "utils\ncalibration, geometry,\nconfig, detection_log"),
        (
            6.25,
            4.8,
            "data_pipeline\nkitti_reader, lidar/image_cropper,\nnormalization, sample_builder,\ndataset, roi_extraction",
        ),
        (10.5, 4.8, "model\nfusion_transformer,\ninference_engine, export, demo"),
        (4.1, 1.5, "training\nlosses, metrics, trainer,\ntrain, evaluate, experiments"),
        (8.4, 1.5, "ros2_nodes\nroi_extraction_node,\ninference_node, kitti_player"),
    ]
    for x, y, t in mods:
        box(ax, x, y, 3.8, 1.7, t, "#e8f1fb", size=10)
    for p, q in [((4.15, 4.8), (4.35, 4.8)), ((8.15, 4.8), (8.6, 4.8)), ((4.1, 2.35), (5.0, 3.95)),
                 ((4.5, 2.35), (9.6, 3.95)), ((8.4, 2.35), (6.9, 3.95)), ((8.8, 2.35), (10.2, 3.95)),
                 ((7.3, 2.35), (2.6, 3.95))]:  # fmt: skip
        arrow(ax, p, q, color="#555555")
    ax.text(6.25, 0.15, "стрілка «A → B»: модуль A використовує модуль B; алгоритми не залежать від ROS 2",
            ha="center", fontsize=10, style="italic")  # fmt: skip
    save(fig, "fig3_modules.png")


# ---------------------------------------------------------------- data


def fig_dataset_stats():
    rows = read_csv(CROPS / "index.csv")
    from multimodal_transformer.data_pipeline.dataset import split_by_scene

    split = split_by_scene([r["sample_id"] for r in rows], 0.2, 42)
    val = set(split["val"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x = np.arange(3)
    tr = [
        sum(1 for r in rows if r["class_name"] == c and r["sample_id"] not in val) for c in CLASSES
    ]
    va = [sum(1 for r in rows if r["class_name"] == c and r["sample_id"] in val) for c in CLASSES]
    axes[0].bar(x - 0.2, tr, 0.4, label="навчальна", color=C_FUSED)
    axes[0].bar(x + 0.2, va, 0.4, label="валідаційна", color=C_LIDAR)
    for i in range(3):
        axes[0].text(i - 0.2, tr[i], str(tr[i]), ha="center", va="bottom", fontsize=9)
        axes[0].text(i + 0.2, va[i], str(va[i]), ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, CLASSES)
    axes[0].set_ylabel("Кількість об'єктів")
    axes[0].set_title("а) розподіл за класами")
    axes[0].legend()
    for c in CLASSES:
        pts = np.array([int(r["num_points"]) for r in rows if r["class_name"] == c])
        dist = np.array([float(r["distance_m"]) for r in rows if r["class_name"] == c])
        bins = np.logspace(np.log10(5), np.log10(max(pts.max(), 6)), 30)
        axes[1].hist(
            pts, bins=bins, histtype="step", lw=1.6, color=C_CLASS[c], label=c, density=True
        )
        axes[2].hist(
            dist,
            bins=np.arange(0, 85, 4),
            histtype="step",
            lw=1.6,
            color=C_CLASS[c],
            label=c,
            density=True,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Кількість точок LiDAR в OBB")
    axes[1].set_title("б) щільність хмар об'єктів")
    axes[2].set_xlabel("Відстань до об'єкта, м")
    axes[2].set_title("в) розподіл за відстанню")
    for a in axes[1:]:
        a.set_ylabel("Щільність розподілу")
        a.legend()
    for a in axes:
        a.grid(alpha=0.3)
    save(fig, "fig2_dataset_stats.png")


def fig_roi_examples():
    rows = read_csv(CROPS / "index.csv")
    picks = []
    rng = np.random.default_rng(7)
    for c in CLASSES:
        cand = [
            r
            for r in rows
            if r["class_name"] == c
            and r["has_image"] == "1"
            and 400 < int(r["num_points"]) < 3000
            and r["occlusion"] == "0"
            and float(r["truncation"]) == 0
            and float(r["distance_m"]) < 15
        ]
        picks.append(cand[int(rng.integers(len(cand)))])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for col, r in enumerate(picks):
        img = cv2.cvtColor(
            cv2.imread(str(CROPS / "images" / f"{r['name']}.png")), cv2.COLOR_BGR2RGB
        )
        pts = np.load(CROPS / "lidar" / f"{r['name']}.npy")
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{r['class_name']}: патч {img.shape[1]}×{img.shape[0]} пкс")
        axes[0, col].axis("off")
        c = pts[:, :3] - pts[:, :3].mean(0)
        sc = axes[1, col].scatter(
            c[:, 1], c[:, 2], s=3, c=pts[:, 3], cmap="viridis", vmin=0, vmax=0.6
        )
        axes[1, col].set_aspect("equal", adjustable="datalim")
        axes[1, col].invert_xaxis()
        axes[1, col].set_xlabel("y, м")
        axes[1, col].set_ylabel("z, м")
        axes[1, col].set_title(f"{len(pts)} точок, {float(r['distance_m']):.1f} м")
        axes[1, col].grid(alpha=0.3)
    fig.tight_layout()
    fig.colorbar(sc, ax=axes[1, :].tolist(), shrink=0.8, pad=0.02, label="Інтенсивність відбиття")
    save(fig, "fig2_roi_examples.png")


def fig_normalization():
    fig, ax = canvas(11, 4.2, (0, 11), (0, 4.4))
    box(ax, 1.4, 2.2, 2.4, 1.2, "Хмара об'єкта\nN × 4\n(x, y, z, i)", "#e3f4e3", size=10)
    box(
        ax,
        4.2,
        2.2,
        2.3,
        1.2,
        "Центрування:\nxyz − центроїд\ni · intensity_scale",
        "#e8f1fb",
        size=10,
    )
    box(ax, 7.0, 3.4, 2.6, 0.9, "N ≥ 1024: випадкова\nпідмножина без повторів", "#fff3d6", size=10)
    box(ax, 7.0, 1.0, 2.6, 0.9, "N < 1024: усі точки +\nвипадкові дублікати", "#fff3d6", size=10)
    box(ax, 9.9, 2.2, 1.8, 1.0, "Тензор\n1024 × 4", "#f3e5f5", size=11)
    arrow(ax, (2.6, 2.2), (3.05, 2.2))
    arrow(ax, (5.35, 2.5), (5.7, 3.3))
    arrow(ax, (5.35, 1.9), (5.7, 1.1))
    arrow(ax, (8.3, 3.3), (9.0, 2.5))
    arrow(ax, (8.3, 1.1), (9.0, 1.9))
    save(fig, "fig2_normalization.png")


# ---------------------------------------------------------------- results


def fig_training_curves():
    h = json.loads((RUN / "training_history.json").read_text(encoding="utf-8"))
    ep = [e["epoch"] for e in h]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(
        ep, [e["train_loss"] for e in h], "-o", color=C_FUSED, label="функція втрат (навчання)"
    )
    ax2 = axes[0].twinx()
    ax2.plot(ep, [e["train_acc"] for e in h], "--s", color=C_LIDAR, label="accuracy (навчання)")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Зважена крос-ентропія")
    ax2.set_ylabel("Accuracy")
    axes[0].set_title("а) навчальна вибірка")
    lines = axes[0].get_lines() + ax2.get_lines()
    axes[0].legend(lines, [line.get_label() for line in lines], loc="center right")
    axes[1].plot(ep, [e["val_macro_f1"] for e in h], "-o", color="black", label="macro F1")
    for c in CLASSES:
        axes[1].plot(
            ep, [e[f"val_f1_{c}"] for e in h], "--", marker=".", color=C_CLASS[c], label=f"F1 {c}"
        )
    best = int(np.argmax([e["val_macro_f1"] for e in h]))
    axes[1].axvline(ep[best], color="#999999", ls=":")
    axes[1].text(ep[best] + 0.1, 0.86, "найкращий\nчекпойнт", fontsize=10)
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("F1")
    axes[1].set_title("б) валідаційна вибірка")
    axes[1].legend(loc="lower right")
    for a in axes:
        a.grid(alpha=0.3)
        a.set_xticks(ep)
    save(fig, "fig3_training_curves.png")


def fig_confusion():
    report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    cm = np.array(report["confusion_fused"])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j] * 100:.1f}%)", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=11)  # fmt: skip
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Істинний клас")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Частка рядка")
    save(fig, "fig4_confusion.png")


def fig_sparse():
    rows = [r for r in read_csv(EXP / "sparse_lidar.csv") if r["points"] != "all"]
    xs = [int(r["points"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(
        xs, [float(r["fused_macro_f1"]) for r in rows], "-o", color=C_FUSED, label="RGB + LiDAR"
    )
    axes[0].plot(
        xs, [float(r["lidar_macro_f1"]) for r in rows], "--s", color=C_LIDAR, label="лише LiDAR"
    )
    axes[0].axhline(0.88, color="#c92a2a", ls=":", lw=1.2)
    axes[0].text(xs[-1] * 1.05, 0.884, "вимога F1 ≥ 0,88", color="#c92a2a", fontsize=10, ha="left")
    axes[0].set_title("а) macro F1")
    for c in CLASSES:
        axes[1].plot(xs, [float(r[f"fused_f1_{c}"]) for r in rows], "-o", color=C_CLASS[c], label=c)
    axes[1].set_title("б) F1 за класами (RGB + LiDAR)")
    for a in axes:
        a.set_xscale("log", base=2)
        a.set_xticks(xs, [str(v) for v in xs])
        a.invert_xaxis()
        a.set_xlabel("Максимальна кількість точок на об'єкт")
        a.set_ylabel("F1")
        a.grid(alpha=0.3)
        a.legend(loc="lower left")
    save(fig, "fig4_sparse_lidar.png")


def fig_breakdowns():
    specs = [
        (
            "distance_m",
            "а) відстань",
            {
                "0-10 m": "0–10",
                "10-20 m": "10–20",
                "20-30 m": "20–30",
                "30-40 m": "30–40",
                ">=40 m": "≥40",
            },
            "м",
        ),
        (
            "num_points",
            "б) точок в OBB",
            {
                "5-15": "5–15",
                "16-63": "16–63",
                "64-255": "64–255",
                "256-1023": "256–1023",
                ">=1024": "≥1024",
            },
            "",
        ),
        (
            "occlusion",
            "в) оклюзія (KITTI)",
            {
                "0 fully visible": "0",
                "1 partly occluded": "1",
                "2 largely occluded": "2",
                "3 unknown": "3",
            },
            "",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (col, title, names, unit) in zip(axes, specs):
        rows = read_csv(EXP / f"errors_by_{col}.csv")
        x = np.arange(len(rows))
        ax.bar(
            x - 0.2,
            [float(r["fused_macro_f1"]) for r in rows],
            0.4,
            color=C_FUSED,
            label="RGB + LiDAR",
        )
        ax.bar(
            x + 0.2,
            [float(r["lidar_macro_f1"]) for r in rows],
            0.4,
            color=C_LIDAR,
            label="лише LiDAR",
        )
        ax.set_xticks(x, [f"{names[r['bin']]}\nn={r['count']}" for r in rows], fontsize=10)
        ax.set_title(title + (f", {unit}" if unit else ""))
        ax.set_ylim(0.5, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Macro F1")
    axes[0].legend(loc="lower left")
    save(fig, "fig4_breakdowns.png")


def fig_speed():
    before = {1: 11.7, 8: 80.9, 32: 3024.7}
    before_mem = {1: 204.5, 8: 1219.2, 32: 4698.6}
    rows = read_csv(EXP / "speed.csv")
    after = {
        int(r["batch"]): float(r["latency_ms"])
        for r in rows
        if r["backend"] == "PyTorch CUDA (FP32)"
    }
    after_mem = {
        int(r["batch"]): float(r["gpu_peak_mem_mb"])
        for r in rows
        if r["backend"] == "PyTorch CUDA (FP32)"
    }
    bs = [1, 8, 32]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(x - 0.2, [before[b] for b in bs], 0.4, color="#adb5bd", label="fast path (до)")
    axes[0].bar(x + 0.2, [after[b] for b in bs], 0.4, color=C_FUSED, label="SDPA (після)")
    axes[1].bar(x - 0.2, [before_mem[b] for b in bs], 0.4, color="#adb5bd", label="fast path (до)")
    axes[1].bar(x + 0.2, [after_mem[b] for b in bs], 0.4, color=C_FUSED, label="SDPA (після)")
    axes[1].axhline(4096, color="#c92a2a", ls=":", lw=1.2)
    axes[1].text(
        1.0, 4500, "обсяг відеопам'яті GTX 1650", color="#c92a2a", fontsize=10, ha="center"
    )
    axes[1].set_ylim(50, 30000)
    axes[0].set_ylim(5, 20000)
    for a, vals_b, vals_a, title, ylab in [
        (axes[0], before, after, "а) затримка прямого проходу", "мс на батч"),
        (axes[1], before_mem, after_mem, "б) пікова пам'ять GPU", "МБ"),
    ]:
        a.set_yscale("log")
        a.set_xticks(x, [f"батч {b}" for b in bs])
        a.set_title(title)
        a.set_ylabel(ylab + " (логарифмічна шкала)")
        a.grid(axis="y", alpha=0.3)
        a.legend(loc="upper left")
        for i, b in enumerate(bs):
            a.text(i - 0.2, vals_b[b] * 1.08, f"{vals_b[b]:.0f}", ha="center", fontsize=9)
            a.text(i + 0.2, vals_a[b] * 1.08, f"{vals_a[b]:.0f}", ha="center", fontsize=9)
    save(fig, "fig4_speed_fix.png")


if __name__ == "__main__":
    fig_fusion_strategies()
    fig_encoder_layer()
    fig_coordinate_chain()
    fig_normalization()
    fig_system_architecture()
    fig_model_architecture()
    fig_ros_graph()
    fig_module_structure()
    fig_dataset_stats()
    fig_roi_examples()
    fig_training_curves()
    fig_confusion()
    fig_sparse()
    fig_breakdowns()
    fig_speed()
