import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ===== 配置 =====
# 两组数据集：(JSON目录, 图片目录)
DATASETS = [
    ("/home/wenyu/Energy/good", "/home/wenyu/Energy/pic"),
    ("/home/wenyu/Energy/good_old", "/home/wenyu/Energy/pic_old"),
]
OUTPUT_DIR = "/home/wenyu/Energy/yolo_dataset"
NUM_KEYPOINTS = 8
CLASSES = {"box": 0, "R": 1, "rect": 2}
VAL_RATIO = 0.2
RANDOM_SEED = 42


def parse_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    # 一个标注文件里可能有多个同名 label（比如多个 rect）
    rectangles = []   # [(label, points), ...]
    keypoints = {}    # corner1-8 -> (x, y)

    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        shape_type = shape["shape_type"]

        if shape_type == "rectangle":
            rectangles.append((label, points))
        elif shape_type == "point" and label.startswith("corner"):
            keypoints[label] = points[0]

    return img_w, img_h, rectangles, keypoints


def rect_to_yolo(pts, img_w, img_h):
    if len(pts) >= 4:
        x1, y1 = pts[0]
        x2, y2 = pts[2]
    elif len(pts) == 2:
        x1, y1 = pts[0]
        x2, y2 = pts[1]
    else:
        return None

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h

    # 检查归一化范围
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None

    return cx, cy, w, h


def convert_one(json_path):
    img_w, img_h, rectangles, keypoints = parse_json(json_path)
    lines = []

    for label, pts in rectangles:
        if label not in CLASSES:
            print(f"[WARN] 未知类别 '{label}'，跳过: {json_path.name}")
            continue

        result = rect_to_yolo(pts, img_w, img_h)
        if result is None:
            continue

        cx, cy, w, h = result
        cls_id = CLASSES[label]

        if label == "box":
            # box 带关键点
            kp_parts = []
            for i in range(1, NUM_KEYPOINTS + 1):
                key = f"corner{i}"
                if key in keypoints:
                    kx = keypoints[key][0] / img_w
                    ky = keypoints[key][1] / img_h
                    # 确保关键点在合理范围内
                    kx = max(0, min(1, kx))
                    ky = max(0, min(1, ky))
                    kp_parts.append(f"{kx:.6f} {ky:.6f} 2")
                else:
                    kp_parts.append("0 0 0")
            kp_str = " ".join(kp_parts)
        else:
            # R 和 rect 无关键点
            kp_str = " ".join(["0 0 0"] * NUM_KEYPOINTS)

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}")

    return lines


def collect_all_samples():
    """从两组数据集收集所有 (json_path, img_path) 对"""
    samples = []

    for json_dir, img_dir in DATASETS:
        json_dir = Path(json_dir)
        img_dir = Path(img_dir)

        if not json_dir.exists():
            print(f"[WARN] JSON 目录不存在: {json_dir}")
            continue
        if not img_dir.exists():
            print(f"[WARN] 图片目录不存在: {img_dir}")
            continue

        json_files = sorted(json_dir.glob("*.json"))
        print(f"数据集 {json_dir.name}: 找到 {len(json_files)} 个标注文件")

        for json_path in json_files:
            stem = json_path.stem

            # 找对应图片
            img_path = None
            for ext in [".jpg", ".png", ".bmp", ".jpeg", ".JPG", ".PNG"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  [WARN] 找不到图片: {stem} (在 {img_dir})")
                continue

            samples.append((json_path, img_path))

    return samples


def build_dataset():
    output = Path(OUTPUT_DIR)

    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 收集所有样本
    samples = collect_all_samples()
    print(f"\n总计 {len(samples)} 个有效样本")

    if len(samples) == 0:
        print("没有找到有效样本，请检查路径")
        return

    # 划分训练集/验证集
    train_samples, val_samples = train_test_split(
        samples, test_size=VAL_RATIO, random_state=RANDOM_SEED
    )

    # 统计类别分布
    class_count = {name: 0 for name in CLASSES}

    for split, split_samples in [("train", train_samples), ("val", val_samples)]:
        valid_count = 0
        skip_count = 0

        for json_path, img_path in split_samples:
            # 避免文件名冲突：用 数据集名_原文件名
            dataset_name = json_path.parent.name
            safe_stem = f"{dataset_name}_{json_path.stem}"

            lines = convert_one(json_path)
            if not lines:
                skip_count += 1
                continue

            # 统计类别
            for line in lines:
                cls_id = int(line.split()[0])
                for name, cid in CLASSES.items():
                    if cid == cls_id:
                        class_count[name] += 1

            # 写标签
            label_path = output / "labels" / split / f"{safe_stem}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(lines))

            # 复制图片（保持扩展名）
            dst_img = output / "images" / split / f"{safe_stem}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)
            valid_count += 1

        print(f"{split}: {valid_count} 张有效, {skip_count} 张跳过")

    # 类别统计
    print(f"\n类别分布:")
    for name, count in class_count.items():
        print(f"  {name} (id={CLASSES[name]}): {count}")

    # 生成 dataset.yaml
    yaml_content = f"""path: {output.resolve()}
train: images/train
val: images/val

kpt_shape: [{NUM_KEYPOINTS}, 3]

names:
  0: box
  1: R
  2: rect
"""
    yaml_path = output / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\ndataset.yaml 已生成: {yaml_path}")
    print("完成!")


if __name__ == "__main__":
    build_dataset()