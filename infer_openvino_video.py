import numpy as np
import cv2
import time
import openvino as ov

# ===== 配置 =====
MODEL_PATH = "runs/power_rune/train_v3_no_rotation/weights/best_int8_openvino_model/best.xml"
VIDEO_PATH = "2026-05-28_10-17-02.avi"
OUTPUT_PATH = "/home/wenyu/Energy/result_video1.avi"

IMGSZ = 480
CONF_THRESH = 0.25
IOU_THRESH = 0.45
NUM_KEYPOINTS = 8
NUM_CLASSES = 2
CLASS_NAMES = {0: "box", 1: "R"}
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
KCONF_THRESH = 0.3
R_CONF_THRESH = 0.6

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
    predictions = output[0].T
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:4 + NUM_CLASSES]
    keypoints_raw = predictions[:, 4 + NUM_CLASSES:]

    max_scores = class_scores.max(axis=1)
    class_ids_all = class_scores.argmax(axis=1)
    # box(0) 用 CONF_THRESH 保证不漏检，R(1) 用更高阈值滤掉误检
    keep = np.where(class_ids_all == 1,
                    max_scores > R_CONF_THRESH,
                    max_scores > CONF_THRESH)
    boxes = boxes[keep]
    class_scores = class_scores[keep]
    keypoints_raw = keypoints_raw[keep]
    max_scores = max_scores[keep]
    class_ids = class_scores.argmax(axis=1)

    if len(boxes) == 0:
        return [], [], [], []

    results_boxes, results_kpts, results_scores, results_classes = [], [], [], []
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
        kpts_orig = [[(kx - dw)/ratio, (ky - dh)/ratio, float(kc)]
                     for kx, ky, kc in kpts]
        results_kpts.append(kpts_orig)

    indices = cv2.dnn.NMSBoxes(
        [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in results_boxes],
        results_scores, CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return [], [], [], []
    indices = indices.flatten()
    return ([results_boxes[i] for i in indices],
            [results_scores[i] for i in indices],
            [results_classes[i] for i in indices],
            [results_kpts[i] for i in indices])


def draw_results(img, boxes, scores, classes, keypoints):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cls_id = classes[i]
        color = COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{CLASS_NAMES[cls_id]} {scores[i]:.2f}",
                    (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls_id == 0:                      # 只有 box 画角点
            valid_pts = []
            for j, (kx, ky, kc) in enumerate(keypoints[i]):
                if kc > KCONF_THRESH:
                    pt = (int(kx), int(ky))
                    valid_pts.append(pt)
                    cv2.circle(img, pt, 4, (255, 0, 255), -1)
                    cv2.putText(img, str(j+1), (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    valid_pts.append(None)
            for a, b in SKELETON:
                if valid_pts[a] and valid_pts[b]:
                    cv2.line(img, valid_pts[a], valid_pts[b], (0, 255, 255), 2)
    return img


def main():
    core = ov.Core()
    model = core.read_model(MODEL_PATH)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (width, height))
    if not writer.isOpened():
        print("VideoWriter 初始化失败!")
        cap.release()
        return

    frame_idx = 0
    total_infer_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_input, ratio, dw, dh = preprocess(frame)
        t0 = time.perf_counter()
        result = infer_request.infer({0: img_input})
        infer_ms = (time.perf_counter() - t0) * 1000
        total_infer_time += infer_ms

        output = result[compiled.output(0)]
        boxes, scores, classes, kpts = postprocess(output, ratio, dw, dh)
        n_box = sum(1 for c in classes if c == 0)
        if n_box == 0:
            print(f"帧 {frame_idx}: 无 box 检出")   # 看漏检有多频繁


        vis = draw_results(frame, boxes, scores, classes, kpts)
        cv2.putText(vis, f"Infer: {infer_ms:.1f}ms  det:{len(boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(vis)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  已处理 {frame_idx}/{total_frames} 帧, "
                  f"平均推理: {total_infer_time/frame_idx:.1f}ms")

    cap.release()
    writer.release()
    avg_ms = total_infer_time / max(frame_idx, 1)
    print(f"\n处理完成! 总帧数: {frame_idx}")
    print(f"平均推理延迟: {avg_ms:.2f} ms ({1000/avg_ms:.0f} FPS)")
    print(f"输出视频: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
