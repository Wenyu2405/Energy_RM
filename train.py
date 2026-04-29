from ultralytics import YOLO

DATASET_YAML = "/home/wenyu/Energy/yolo_dataset/dataset.yaml"
IMGSZ = 480
EPOCHS = 300
BATCH = 16


def main():
    model = YOLO("yolov8n-pose.pt")

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        box=7.5,
        cls=1.0,
        pose=12.0,
        kobj=1.0,

        mosaic=0.6,
        close_mosaic=20,
        mixup=0.1,
        degrees=180.0,
        translate=0.1,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        dropout=0.0,
        patience=50,
        save_period=20,
        device=0,
        workers=8,
        project="runs/power_rune",
        name="train_v1",
    )

    # 用 results.save_dir 获取实际输出路径，不要写死
    best_pt = f"{results.save_dir}/weights/best.pt"
    print(f"最佳模型路径: {best_pt}")

    best = YOLO(best_pt)
    best.export(format="openvino", imgsz=IMGSZ, half=True, simplify=True)

    print("训练完成，OpenVINO FP16 模型已导出")


if __name__ == "__main__":
    main()