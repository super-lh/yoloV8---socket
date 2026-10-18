# -*- coding: utf-8 -*-
"""
把标注框画到图片上，方便人工核对标注有没有画歪

用法：
    python preview_labels.py                 # 生成全部预览
    python preview_labels.py --split val     # 只看验证集

输出目录：<数据集根目录>\\preview\\<split>\\*.jpg（跟随 BASE，当前为 D:\\super-lhyolov8socket\\yoloV8---socket\\dataset\\preview）
打开图片看一眼，框应该正好贴在插座边缘上；没有框的图会标 "no box (negative)"。
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np

# 【关键】把工作目录固定到脚本所在目录 D:\YOLOv8\training
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

BASE = r"D:\super-lhyolov8socket\yoloV8---socket\dataset"  # 2026-10-18 修正原路径 D:\YOLOv8\dataset 已不存在
OUT = os.path.join(os.path.dirname(BASE), "preview")  # 2026-10-18：dataset 内只保留 images/labels，预览输出到工程根目录 preview/
MAX_SIDE = 900


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--base", default=BASE)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="出错时返回非零退出码（供脚本串联）；不加则始终返回 0，调试时不会中断",
    )
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        idir = os.path.join(args.base, "images", split)
        ldir = os.path.join(args.base, "labels", split)
        odir = os.path.join(OUT, split)
        os.makedirs(odir, exist_ok=True)
        if not os.path.isdir(idir):
            continue
        n = 0
        n_img = len([f for f in os.listdir(idir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        if n_img == 0:
            print("[%s] 目录里没有图片: %s" % (split, idir))
            continue
        for fn in sorted(os.listdir(idir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            # 用 imdecode 读，避免 Windows 下中文路径读不了
            data = open(os.path.join(idir, fn), "rb").read()
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("读不了:", fn)
                continue
            h, w = img.shape[:2]

            txt = os.path.join(ldir, stem + ".txt")
            boxes = []
            if os.path.exists(txt):
                for ln in open(txt, encoding="utf-8", errors="replace").read().splitlines():
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    try:
                        cls = int(float(p[0]))
                        cx, cy, bw, bh = [float(v) for v in p[1:5]]
                    except ValueError:
                        continue
                    boxes.append((cls, cx, cy, bw, bh))

            for cls, cx, cy, bw, bh in boxes:
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    img,
                    "socket",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if not boxes:
                cv2.putText(
                    img,
                    "no box (negative)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            scale = min(1.0, MAX_SIDE / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imwrite(os.path.join(odir, stem + ".jpg"), img)
            n += 1
        print("[%s] 生成 %d 张预览 -> %s" % (split, n, odir))

    print()
    print("打开文件夹看：%s" % OUT)
    return 0


if __name__ == "__main__":
    code = main()
    if code and "--strict" in sys.argv:
        raise SystemExit(code)
