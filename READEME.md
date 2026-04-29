# Energy - 能量机关检测

基于 YOLOv8 的能量机关（Power Rune）检测模型，支持 OpenVINO INT8 量化部署。

## 项目结构


Energy/
├──good #标注文件
├──pic #照片
├──good_old #旧的标注文件（可不用）
├──good_old #新的标注文件 （可不用）
├── datasets/
│ └── power_rune/ # 原始数据集
├── yolo_dataset/ # YOLO 格式数据集
│ ├── images/
│ │ ├── train/
│ │ └── val/ # 1523 张验证图片
│ └── labels/
│ ├── train/
│ └── val/
├── runs1/power_rune/train_v12/
│ └── weights/
│ ├── best.pt # PyTorch 原始权重
│ ├── best_openvino_model/ # FP16 OpenVINO 模型
│ └── best_int8_openvino_model/ # INT8 OpenVINO 模型
├── quantize.py # INT8 量化脚本
├── deploy.py # FP16 vs INT8 对比测试
├── debug_output_v2.py # 模型输出调试工具
└── test_results/ # 对比可视化结果


## 模型参数

| 参数 | 值 |
|------|-----|
| 模型架构 | YOLOv8 |
| 输入尺寸 | 480 × 480 |
| 数据集 | power_rune（3175 实例） |
| 训练版本 | train_v12 |

## 性能指标

### 精度（验证集 1523 张图片）

| 模型 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| FP16 | 0.9948 | 0.9048 | 0.999 | 0.999 |
| INT8 | 0.9933 | 0.7822 | 0.945 | 0.996 |
| 差异 | -0.15% | -13.5% | -5.4% | -0.3% |

### 速度（CPU，OpenVINO LATENCY 模式）

| 模型 | 单张推理 | 批量推理 | 模型大小 |
|------|---------|---------|---------|
| FP16 | 51.4 ms | 17.6 ms | 6.0 MB |
| INT8 | 38.4 ms | 6.0 ms | 3.1 MB |
| 加速比 | 1.34x | 2.93x | 1.96x 压缩 |

## 环境依赖

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- Ultralytics 8.3.85
- OpenVINO 2024.6.0
- NNCF（用于 INT8 量化）

### 安装

```bash
pip install ultralytics
pip install openvino==2024.6.0
pip install nncf
使用方法
训练
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="datasets/power_rune/data.yaml", imgsz=480, epochs=100)
导出 OpenVINO FP16
python3 -c "
from ultralytics import YOLO
model = YOLO('runs1/power_rune/train_v12/weights/best.pt')
model.export(format='openvino', imgsz=480, half=True)
"
INT8 量化
python3 quantize.py

量化完成后将 metadata.yaml 复制到 INT8 模型目录：

cp runs1/power_rune/train_v12/weights/best_openvino_model/metadata.yaml \
   runs1/power_rune/train_v12/weights/best_int8_openvino_model/
精度与速度测试
python3 deploy.py

测试结果保存在 test_results/ 目录，包含 FP16 vs INT8 的可视化对比图。

推理
from ultralytics import YOLO

# PyTorch（GPU）
model = YOLO("runs1/power_rune/train_v12/weights/best.pt")
results = model("your_image.jpg", imgsz=480)

# OpenVINO FP16（CPU）
model = YOLO("runs1/power_rune/train_v12/weights/best_openvino_model", task="detect")
results = model("your_image.jpg", imgsz=480)

# OpenVINO INT8（CPU，推荐部署用）
model = YOLO("runs1/power_rune/train_v12/weights/best_int8_openvino_model", task="detect")
results = model("your_image.jpg", imgsz=480)

# 处理结果
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()   # 边界框坐标
    confs = r.boxes.conf.cpu().numpy()   # 置信度
    classes = r.boxes.cls.cpu().numpy()  # 类别
量化说明

INT8 量化使用 NNCF（Neural Network Compression Framework），基于 300 张校准图片进行统计量收集和偏差校正。

mAP50 仅下降 0.15%，检测框数量完全一致，适合直接部署。mAP50-95 下降 13.5% 是因为高 IoU 阈值下边界框坐标精度有损失，实际使用中影响较小。

硬件环境

GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)

OS: Ubuntu 22.04

CPU: 用于 OpenVINO 推理测试