import numpy as np
import cv2
from pathlib import Path
import openvino as ov
import nncf

FP16_MODEL = "/home/wenyu/Energy/runs1/power_rune/train_v12/weights/best_openvino_model/best.xml"
CALIB_DIR = "yolo_dataset/images/val"
OUTPUT_DIR = "runs1/power_rune/train_v12/weights/best_int8_openvino_model"
IMGSZ = 480
SUBSET_SIZE = 300


def letterbox(img, new_shape=480):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = new_shape - new_w
    dh = new_shape - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def preprocess(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = letterbox(img, IMGSZ)
    # 不做BGR→RGB转换
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img


def main():
    core = ov.Core()
    ov_model = core.read_model(FP16_MODEL)
    print(f"模型输入: {ov_model.input().shape}")

    img_paths = sorted(Path(CALIB_DIR).glob("*.*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg", ".png", ".bmp", ".jpeg"}]
    img_paths = img_paths[:SUBSET_SIZE]
    print(f"校准图片数量: {len(img_paths)}")

    calibration_dataset = nncf.Dataset(img_paths, preprocess)

    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        subset_size=len(img_paths),
        preset=nncf.QuantizationPreset.MIXED,
        fast_bias_correction=False,
    )

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized_model, str(output_path / "best.xml"))

    fp16_bin = Path(FP16_MODEL).with_suffix(".bin")
    int8_bin = output_path / "best.bin"
    if fp16_bin.exists() and int8_bin.exists():
        fp16_size = fp16_bin.stat().st_size / 1024 / 1024
        int8_size = int8_bin.stat().st_size / 1024 / 1024
        print(f"\nFP16: {fp16_size:.1f} MB")
        print(f"INT8: {int8_size:.1f} MB")
        print(f"压缩比: {fp16_size/int8_size:.2f}x")

    print(f"\nINT8 模型已保存到: {output_path}")


if __name__ == "__main__":
    main()