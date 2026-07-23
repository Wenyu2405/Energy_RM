# Energy - 能量机关检测（YOLOv8-Pose）

基于 YOLOv8-Pose 的能量机关（Power Rune）检测模型。相比旧版的纯目标检测，
新版升级为姿态估计，在检测框和类别之外额外回归 8 个角点关键点，供后续 PnP 位姿解算使用。
支持 OpenVINO FP16 导出与 INT8 量化，CPU 端实时推理。

## 版本变化总览（旧版 → 新版）

| 项目 | 旧版 | 新版 |
|------|------|------|
| 任务类型 | 目标检测（detect） | 姿态估计（pose，检测框 + 8 角点） |
| 模型架构 | YOLOv8 | YOLOv8s-Pose |
| 类别 | 3 类（box / R / rect） | 2 类（box / R） |
| 模型输出通道 | 31（4 + 3 + 24） | 30（4 + 2 + 8×3） |
| 角点关键点 | 无 | 8 个（含 x / y / conf） |
| 视频推理 | 无完整管线 | 完整视频推理脚本 |
| 用途 | 仅检测框 | 角点输出供 PnP 解算 |

### 1. 从检测升级为姿态估计
旧版只输出检测框和类别。新版改用 YOLOv8s-Pose，额外回归扇叶的 8 个角点，
每个角点包含坐标和置信度（已过 sigmoid，取值 [0,1]）。角点坐标在后处理中
还原到原图坐标系，可直接送 PnP 解算求解装甲板位姿。

### 2. 类别精简：3 类 → 2 类
旧版含 `box` / `R` / `rect` 三类。新版删除不再使用的 `rect`，仅保留
`box`（待激活扇叶）和 `R`（中心标）两类。相应地：
- `Aconvert_to_yolo_pose.py` 的类别定义和生成的 `dataset.yaml` 中 `names` 同步为 2 类
- 推理端 `NUM_CLASSES` 改为 2，模型输出通道从 31 变为 30

**注意：旧的 3 类模型（31 通道）与新代码（按 2 类切分）不兼容，混用会导致关键点解析
错位（reshape 报错）。必须用新数据重新训练，并重新量化 INT8 模型。**

### 3. 数据集：新老角点对齐
合并了新老两批标注数据。老数据集的角点标注顺序与新数据集存在固定循环偏移，
在 `Aconvert_to_yolo_pose.py` 中通过 `REMAP_OLD` 做角点编号重映射，保证同一
物理角点在新老数据中编号一致，避免角点监督信号冲突。

### 4. 量化预处理对齐
`quantize.py` 的校准预处理与训练/推理严格对齐：居中 letterbox + BGR→RGB + 归一化，
避免预处理不一致引入量化偏差。量化后需将 `metadata.yaml` 复制到 INT8 模型目录，
否则 ultralytics 无法自动识别为 pose 任务。

### 5. 新增完整视频推理管线
新增 `infer_openvino_video.py`，直接用 OpenVINO 原生 API 推理，逐帧检测并绘制
检测框、类别、8 角点及骨架连线，输出可视化视频。

## 项目结构


Energy/
├── Aconvert_to_yolo_pose.py # 原始标注 → YOLO-Pose 格式（顺时针），含新老数据集角点对齐
├── convert_to_yolo_pose.py # 角点转换（逆时针版本）
├── train.py # 训练 + 导出 OpenVINO
├── quantize.py # INT8 量化（NNCF）
├── deploy.py # FP16 vs INT8 精度/速度对比
├── infer_openvino_video.py # 视频推理（逐帧检测 + 角点可视化）
├── check_labels.py # 标注可视化核对工具
├── good/ # 新标注文件
├── good_old/ # 旧标注文件
├── pic/ # 图片
├── yolo_dataset/ # 转换后的数据集（自动生成）
└── test_results/ # FP16 vs INT8 对比可视化


## 模型参数

| 参数 | 值 |
|------|-----|
| 模型架构 | YOLOv8s-Pose |
| 任务 | pose（检测框 + 8 角点） |
| 输入尺寸 | 480 × 480 |
| 类别数 | 2（box / R） |
| 关键点数 | 8（每点 x / y / conf） |
| 输出形状 | (1, 30, N) |
| 训练版本 | train_v3_no_rotation |

## 性能指标

### 精度（验证集 198 张 / 396 实例）

| 指标 | FP16 | INT8 |
|------|------|------|
| Box mAP50 | 0.9950 | 0.9933 |
| Box mAP50-95 | 0.9428 | 0.7168 |
| Pose mAP50（box 类） | 0.995 | 0.995 |

> Box mAP50-95 在 INT8 下下降较多，是高 IoU 阈值下边界框亚像素精度损失所致；
> Box mAP50 几乎无变化，且关键点精度（Pose mAP50）量化前后完全一致（均为 0.995），
> 对角点解算无影响。

### 速度（CPU: i5-13500H, OpenVINO LATENCY 模式）

| 模型 | 单张推理 | 模型大小 |
|------|----------|----------|
| FP16 | ~60 ms | ~22 MB |
| INT8 | ~26 ms | ~11 MB |
| 加速比 | 2.33x | 1.99x 压缩 |

视频推理（1280×1024，OpenVINO 原生 API）约 13 ms/帧，77 FPS。

## 环境依赖


Python 3.10
PyTorch 2.6.0 + CUDA 12.4
Ultralytics 8.3.85
OpenVINO 2024.6.0
NNCF（用于 INT8 量化）
opencv-python
numpy


```bash
pip install ultralytics
pip install openvino==2024.6.0
pip install nncf
使用流程
# 1. 数据转换（原始标注 → YOLO-Pose，含新老角点对齐）
python3 Aconvert_to_yolo_pose.py

# 2. （可选）核对标注是否正确
python3 check_labels.py

# 3. 训练并导出 OpenVINO FP16
python3 train.py

# 4. INT8 量化
python3 quantize.py

# 5. 复制 metadata（让 INT8 模型可被识别为 pose 任务）
cp runs/power_rune/train_v3_no_rotation/weights/best_openvino_model/metadata.yaml \
   runs/power_rune/train_v3_no_rotation/weights/best_int8_openvino_model/

# 6. FP16 vs INT8 精度速度对比
python3 deploy.py

# 7. 视频推理可视化
python3 infer_openvino_video.py

各脚本顶部的路径变量（模型路径、视频路径、数据集路径）请按本地实际情况修改。

推理输出说明

模型输出形状 (1, 30, N)，其中 30 = 4（框 xywh） + 2（类别） + 8×3（角点 x/y/conf）。
后处理流程：置信度过滤 → 坐标还原到原图 → NMS。角点置信度已过 sigmoid，
默认过滤阈值 0.3。角点坐标输出为原图像素坐标，可直接用于 PnP 解算。

推理端关键阈值：

CONF_THRESH：box 检测阈值，建议 0.25（过高会导致逐帧漏检、可视化闪烁）

R_CONF_THRESH：R 标单独阈值，建议 0.6（滤除低置信度误检）

KCONF_THRESH：角点显示阈值，0.3

已知问题 / 说明

验证集指标接近饱和（Pose mAP50 ≈ 0.995），但与训练集分布接近，实车表现应以
实拍视频为准。

视频推理为纯逐帧输出，不做跨帧平滑，以便真实评估模型质量；角点滤波交由
下游 PnP 解算端处理。

CONF_THRESH 设置过高（如 0.5）会导致部分帧漏检、可视化时框闪烁，建议 0.25。

硬件环境

GPU: NVIDIA GeForce RTX 4050 Laptop (6GB)

CPU: 13th Gen Intel Core i5-13500H（用于 OpenVINO 推理测试）

OS: Ubuntu 22.04