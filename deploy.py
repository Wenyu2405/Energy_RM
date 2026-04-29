"""
test_int8_vs_fp16.py
对比 FP16 和 INT8 OpenVINO 模型的精度与速度
"""

import os
import time
import glob
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 配置
# ============================================================
FP16_MODEL = "runs1/power_rune/train_v12/weights/best_openvino_model"
INT8_MODEL = "runs1/power_rune/train_v12/weights/best_int8_openvino_model"

# 测试图片目录，改成你自己的验证集路径
TEST_IMG_DIR = "yolo_dataset/images/val"
# 如果没有验证集，也可以指定单张图片
# TEST_IMG_DIR = "test_image.jpg"

OUTPUT_DIR = "test_results"
CONF_THRESH = 0.25
IOU_THRESH = 0.45
IMG_SIZE = 480
WARMUP_RUNS = 5
SPEED_RUNS = 50


def load_test_images(path, max_images=50):
    """加载测试图片路径列表"""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for ext in exts:
        images.extend(glob.glob(str(p / ext)))
    images.sort()
    if len(images) > max_images:
        step = len(images) // max_images
        images = images[::step][:max_images]
    return images


def test_with_ultralytics():
    """方法一：用 ultralytics API 测试（推荐）"""
    from ultralytics import YOLO

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = load_test_images(TEST_IMG_DIR)
    print(f"测试图片数量: {len(images)}")

    if len(images) == 0:
        print(f"错误: 在 {TEST_IMG_DIR} 中没有找到图片")
        print("请修改 TEST_IMG_DIR 变量指向你的测试图片目录")
        return

    # ----------------------------------------------------------
    # 加载模型
    # ----------------------------------------------------------
    print("\n加载 FP16 模型...")
    model_fp16 = YOLO(FP16_MODEL)

    print("加载 INT8 模型...")
    model_int8 = YOLO(INT8_MODEL)

    # ----------------------------------------------------------
    # 速度测试
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("速度测试")
    print("=" * 60)

    test_img = images[0]

    # warmup
    for _ in range(WARMUP_RUNS):
        model_fp16(test_img, imgsz=IMG_SIZE, verbose=False)
        model_int8(test_img, imgsz=IMG_SIZE, verbose=False)

    # FP16 速度
    t0 = time.time()
    for _ in range(SPEED_RUNS):
        model_fp16(test_img, imgsz=IMG_SIZE, verbose=False)
    fp16_time = (time.time() - t0) / SPEED_RUNS * 1000

    # INT8 速度
    t0 = time.time()
    for _ in range(SPEED_RUNS):
        model_int8(test_img, imgsz=IMG_SIZE, verbose=False)
    int8_time = (time.time() - t0) / SPEED_RUNS * 1000

    print(f"FP16 平均推理时间: {fp16_time:.1f} ms")
    print(f"INT8 平均推理时间: {int8_time:.1f} ms")
    print(f"加速比: {fp16_time / int8_time:.2f}x")

    # ----------------------------------------------------------
    # 精度对比
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("精度对比")
    print("=" * 60)

    fp16_total_boxes = 0
    int8_total_boxes = 0
    fp16_confs = []
    int8_confs = []
    match_count = 0
    mismatch_count = 0

    for i, img_path in enumerate(images):
        r_fp16 = model_fp16(img_path, imgsz=IMG_SIZE, conf=CONF_THRESH,
                            iou=IOU_THRESH, verbose=False)[0]
        r_int8 = model_int8(img_path, imgsz=IMG_SIZE, conf=CONF_THRESH,
                            iou=IOU_THRESH, verbose=False)[0]

        n_fp16 = len(r_fp16.boxes)
        n_int8 = len(r_int8.boxes)
        fp16_total_boxes += n_fp16
        int8_total_boxes += n_int8

        if n_fp16 > 0:
            fp16_confs.extend(r_fp16.boxes.conf.cpu().numpy().tolist())
        if n_int8 > 0:
            int8_confs.extend(r_int8.boxes.conf.cpu().numpy().tolist())

        if n_fp16 == n_int8:
            match_count += 1
        else:
            mismatch_count += 1

        # 保存前 10 张对比图
        if i < 10:
            save_comparison(img_path, r_fp16, r_int8, i)

    print(f"\n{'指标':<25} {'FP16':<15} {'INT8':<15}")
    print("-" * 55)
    print(f"{'总检测框数':<25} {fp16_total_boxes:<15} {int8_total_boxes:<15}")

    fp16_avg = np.mean(fp16_confs) if fp16_confs else 0
    int8_avg = np.mean(int8_confs) if int8_confs else 0
    print(f"{'平均置信度':<25} {fp16_avg:<15.4f} {int8_avg:<15.4f}")

    fp16_med = np.median(fp16_confs) if fp16_confs else 0
    int8_med = np.median(int8_confs) if int8_confs else 0
    print(f"{'中位数置信度':<25} {fp16_med:<15.4f} {int8_med:<15.4f}")

    print(f"{'检测数一致的图片':<25} {match_count}/{len(images)}")
    print(f"{'检测数不一致的图片':<25} {mismatch_count}/{len(images)}")

    if fp16_total_boxes > 0:
        box_diff = abs(fp16_total_boxes - int8_total_boxes) / fp16_total_boxes * 100
        print(f"{'检测框数差异':<25} {box_diff:.1f}%")

    # ----------------------------------------------------------
    # 在验证集上跑 mAP（如果有标注）
    # ----------------------------------------------------------
    data_yaml = find_data_yaml()
    if data_yaml:
        print("\n" + "=" * 60)
        print("mAP 评估（验证集）")
        print("=" * 60)

        print("\nFP16 mAP:")
        metrics_fp16 = model_fp16.val(data=data_yaml, imgsz=IMG_SIZE, verbose=False)
        print(f"  mAP50:    {metrics_fp16.box.map50:.4f}")
        print(f"  mAP50-95: {metrics_fp16.box.map:.4f}")

        print("\nINT8 mAP:")
        metrics_int8 = model_int8.val(data=data_yaml, imgsz=IMG_SIZE, verbose=False)
        print(f"  mAP50:    {metrics_int8.box.map50:.4f}")
        print(f"  mAP50-95: {metrics_int8.box.map:.4f}")

        map_drop = (metrics_fp16.box.map50 - metrics_int8.box.map50) / metrics_fp16.box.map50 * 100
        print(f"\nmAP50 下降: {map_drop:.2f}%")
        if abs(map_drop) < 1:
            print("量化质量: 优秀（下降 < 1%）")
        elif abs(map_drop) < 3:
            print("量化质量: 良好（下降 < 3%）")
        else:
            print("量化质量: 需要关注（下降 >= 3%）")
    else:
        print(f"\n未找到 data.yaml，跳过 mAP 评估")
        print("如需 mAP 评估，请确保 data.yaml 路径正确")

    print(f"\n对比图片已保存到: {OUTPUT_DIR}/")


def save_comparison(img_path, r_fp16, r_int8, idx):
    """保存 FP16 vs INT8 对比图"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img_fp16 = img.copy()
    img_int8 = img.copy()

    # 画 FP16 结果
    if len(r_fp16.boxes) > 0:
        for box, conf, cls in zip(r_fp16.boxes.xyxy.cpu().numpy(),
                                   r_fp16.boxes.conf.cpu().numpy(),
                                   r_fp16.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_fp16, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(img_fp16, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 画 INT8 结果
    if len(r_int8.boxes) > 0:
        for box, conf, cls in zip(r_int8.boxes.xyxy.cpu().numpy(),
                                   r_int8.boxes.conf.cpu().numpy(),
                                   r_int8.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_int8, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(img_int8, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 添加标题
    cv2.putText(img_fp16, "FP16", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_int8, "INT8", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 拼接
    combined = np.hstack([img_fp16, img_int8])

    # 如果太大就缩放
    max_width = 1920
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"compare_{idx:03d}.jpg")
    cv2.imwrite(save_path, combined)


def find_data_yaml():
    """尝试找到 data.yaml"""
    candidates = [
        "datasets/power_rune/data.yaml",
        "datasets/power_rune.yaml",
        "data/power_rune.yaml",
        "data.yaml",
        "runs1/power_rune/train_v12/args.yaml",
    ]
    for c in candidates:
        if os.path.exists(c):
            # 如果是 args.yaml，需要从中提取 data 路径
            if "args.yaml" in c:
                import yaml
                with open(c) as f:
                    args = yaml.safe_load(f)
                if "data" in args and os.path.exists(args["data"]):
                    return args["data"]
            else:
                return c
    return None


if __name__ == "__main__":
    test_with_ultralytics()