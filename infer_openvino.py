import numpy as np
import cv2
import time
import openvino as ov

# ===== 配置 =====
MODEL_PATH = "/home/wenyu/Energy/runs1/power_rune/train_v1/weights/best_int8_openvino_model/best.xml"
IMAGE_PATH = "/home/wenyu/Energy/yolo_dataset/images/val/good_Image186.jpg"

IMGSZ = 480
CONF_THRESH = 0.5
IOU_THRESH = 0.45
NUM_KEYPOINTS = 8
NUM_CLASSES = 3
CLASS_NAMES = {0: "box", 1: "R", 2: "rect"}
SKELETON = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)]
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}


def letterbox(img, new_shape=480):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, r, (dw, dh)


def preprocess(img):
    img_lb, ratio, (dw, dh) = letterbox(img, IMGSZ)
    img_input = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...]
    return img_input, ratio, dw, dh


def postprocess(output, ratio, dw, dh):
    """
    YOLOv8-pose 输出: [1, 4+num_classes+num_kpts*3, num_anchors]
    3类8关键点: [1, 4+3+24, num_anchors] = [1, 31, num_anchors]
    """
    predictions = output[0].T  # [num_anchors, 31]

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:4+NUM_CLASSES]
    keypoints_raw = predictions[:, 4+NUM_CLASSES:]

    max_scores = class_scores.max(axis=1)
    mask = max_scores > CONF_THRESH
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    keypoints_raw = keypoints_raw[mask]
    max_scores = max_scores[mask]
    class_ids = class_scores.argmax(axis=1)

    if len(boxes) == 0:
        return [], [], [], []

    # 转换到原图坐标
    results_boxes = []
    results_kpts = []
    results_scores = []
    results_classes = []

    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        x1 = (cx - w/2 - dw) / ratio
        y1 = (cy - h/2 - dh) / ratio
        x2 = (cx + w/2 - dw) / ratio
        y2 = (cy + h/2 - dh) / ratio

        results_boxes.append([x1, y1, x2, y2])
        results_scores.append(float(max_scores[i]))
        results_classes.append(int(class_ids[i]))

        kpts = keypoints_raw[i].reshape(NUM_KEYPOINTS, 3)
        kpts_orig = []
        for kx, ky, kconf in kpts:
            orig_x = (kx - dw) / ratio
            orig_y = (ky - dh) / ratio
            kpts_orig.append([orig_x, orig_y, float(kconf)])
        results_kpts.append(kpts_orig)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in results_boxes],
        results_scores, CONF_THRESH, IOU_THRESH
    )

    if len(indices) == 0:
        return [], [], [], []

    indices = indices.flatten()
    return (
        [results_boxes[i] for i in indices],
        [results_scores[i] for i in indices],
        [results_classes[i] for i in indices],
        [results_kpts[i] for i in indices],
    )


def draw_results(img, boxes, scores, classes, keypoints):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cls_id = classes[i]
        score = scores[i]
        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 只对 box 类画关键点
        if cls_id == 0:
            kpts = keypoints[i]
            valid_pts = []
            for j, (kx, ky, kconf) in enumerate(kpts):
                if kconf > 0.3:
                    pt = (int(kx), int(ky))
                    valid_pts.append(pt)
                    cv2.circle(img, pt, 4, (255, 0, 255), -1)
                    cv2.putText(img, str(j+1), (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    valid_pts.append(None)

            for a, b in SKELETON:
                if valid_pts[a] is not None and valid_pts[b] is not None:
                    cv2.line(img, valid_pts[a], valid_pts[b], (0, 255, 255), 2)

    return img


def main():
    core = ov.Core()
    model = core.read_model(MODEL_PATH)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"无法读取图片: {IMAGE_PATH}")
        return

    img_input, ratio, dw, dh = preprocess(img)

    # Warmup
    for _ in range(10):
        infer_request.infer({0: img_input})

    # 计时
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        result = infer_request.infer({0: img_input})
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / N * 1000
    print(f"平均推理延迟: {avg_ms:.2f} ms ({1000/avg_ms:.0f} FPS)")

    # 后处理
    output = result[compiled.output(0)]
    boxes, scores, classes, kpts = postprocess(output, ratio, dw, dh)
    print(f"检测到 {len(boxes)} 个目标")
    for i in range(len(boxes)):
        print(f"  {CLASS_NAMES[classes[i]]}: {scores[i]:.3f}")

    # 可视化
    vis = draw_results(img.copy(), boxes, scores, classes, kpts)
    cv2.imwrite("result.jpg", vis)
    print("结果已保存到 result.jpg")


if __name__ == "__main__":
    main()
