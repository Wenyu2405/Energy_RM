"""核对 yolo_dataset 里角点标注是否正确、新老数据集角点语义是否一致。
用法: python check_labels.py
看输出图：同编号(同颜色)的角点在 good_ 和 good_old_ 开头的图里
是否都落在扇叶的同一物理位置。不一致就说明 REMAP_OLD 偏移错了。
"""
from pathlib import Path
import cv2

IMG_DIR = Path("/home/wenyu/Energy/yolo_dataset/images/train")
LBL_DIR = Path("/home/wenyu/Energy/yolo_dataset/labels/train")
OUT_DIR = Path("/home/wenyu/Energy/label_viz")
N_KPTS = 8
LIMIT = 60

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 128, 255), (255, 128, 0)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in IMG_DIR.iterdir()
                  if p.suffix.lower() in (".jpg", ".png", ".bmp", ".jpeg"))
    # 新老各取一些
    old = [p for p in imgs if p.name.startswith("good_old")][:LIMIT // 2]
    new = [p for p in imgs if not p.name.startswith("good_old")][:LIMIT // 2]
    for p in old + new:
        lbl = LBL_DIR / (p.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for line in lbl.read_text().strip().splitlines():
            v = list(map(float, line.split()))
            cls = int(v[0])
            cx, cy, bw, bh = v[1:5]
            x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
            x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(img, f"c{cls}", (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if cls == 0:   # 只有 box 有角点
                kp = v[5:]
                for i in range(N_KPTS):
                    kx, ky = int(kp[i*3]*w), int(kp[i*3+1]*h)
                    if kx <= 0 and ky <= 0:
                        continue
                    cv2.circle(img, (kx, ky), 4, COLORS[i], -1)
                    cv2.putText(img, str(i+1), (kx+4, ky),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[i], 1)
        cv2.imwrite(str(OUT_DIR / p.name), img)
    print(f"输出到 {OUT_DIR}，逐张对比 good_ 和 good_old_ 开头的图")


if __name__ == "__main__":
    main()