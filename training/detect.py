# -*- coding: utf-8 -*-
"""
插座检测程序（最简版：给文件就出结果）

用法（推荐把文件或文件夹拖到 拖到这里检测.bat 上，连命令都不用敲）：

    python detect.py 图片.jpg
    python detect.py 视频.mp4
    python detect.py 图片文件夹
    python detect.py 0                 # 0 = 摄像头实时
    python detect.py 图片.jpg --conf 0.25

做的事：
  1. 自动找 runs/ 下最新训练出来的 best.pt（也可用 --weights 指定）
  2. 逐张/逐帧检测，在画面上画框
  3. 结果保存到 本脚本所在目录的 outputs\\（当前为 D:\\super-lhyolov8socket\\yoloV8---socket\\training\\outputs）
  4. 终端里打印每个框的类别、置信度、坐标
  5. 图片/视频处理完自动打开结果（加 --no-open 可关闭）

快捷键（摄像头或视频播放窗口）：q 或 ESC 退出，s 保存当前画面
"""
import argparse
import glob
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

# 【关键】工作目录固定到脚本所在目录 D:\YOLOv8\training
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT_DIR = os.path.join(_SCRIPT_DIR, "outputs")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_latest_best():
    cands = []
    for pat in ("runs/**/best.pt", "runs/**/last.pt"):
        cands += glob.glob(pat, recursive=True)
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return os.path.normpath(cands[0])


def open_path(p):
    try:
        os.startfile(p)  # Windows：用默认程序打开
    except Exception:
        pass


def describe(result, model):
    """把一帧的检测结果整理成文字"""
    boxes = getattr(result, "boxes", None)
    n = len(boxes) if boxes is not None else 0
    names = model.names
    lines = []
    if boxes is not None:
        for i in range(n):
            cid = int(boxes.cls[i])
            name = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            conf = float(boxes.conf[i])
            xyxy = [round(float(v), 1) for v in boxes.xyxy[i].tolist()]
            lines.append("   %s  %.2f  坐标 %s" % (name, conf, xyxy))
    return n, lines


def do_image(model, path, conf, imgsz, do_open):
    data = open(path, "rb").read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("  读不了这张图:", path)
        return None
    res = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = res.plot()
    n, lines = describe(res, model)
    os.makedirs(OUT_DIR, exist_ok=True)
    save = os.path.join(OUT_DIR, "detect_" + os.path.basename(path))
    cv2.imwrite(save, out)
    print("  %s -> 检出 %d 个" % (os.path.basename(path), n))
    for ln in lines:
        print(ln)
    print("  已保存:", save)
    if do_open:
        open_path(save)
    return save


def do_video(model, path, conf, imgsz, do_open, show):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  打不开视频:", path)
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save = os.path.join(OUT_DIR, "detect_" + os.path.basename(path))
    if not save.lower().endswith(".mp4"):
        save = os.path.splitext(save)[0] + ".mp4"
    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    t0 = time.time()
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        res = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        out = res.plot()
        n, _ = describe(res, model)
        total += n
        writer.write(out)
        if show:
            cv2.putText(out, "frame %d | %d sockets" % (idx, n), (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("socket detector", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                p = os.path.join(OUT_DIR, "snap_%d.jpg" % int(time.time()))
                cv2.imwrite(p, out)
                print("  已截图:", p)
        elif idx % 20 == 0:
            print("  已处理 %d 帧, 累计检出 %d 个" % (idx, total))
    writer.release()
    cap.release()
    if show:
        cv2.destroyAllWindows()
    print("  共 %d 帧, 累计检出 %d 个, 用时 %.1f 秒" % (idx, total, time.time() - t0))
    print("  已保存:", save)
    if do_open:
        open_path(save)
    return save


def do_camera(model, conf, imgsz):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("打不开摄像头")
        return 1
    print("摄像头已开启：q 或 ESC 退出，s 截图")
    frames = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取失败")
            break
        frames += 1
        res = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        out = res.plot()
        n, _ = describe(res, model)
        fps = frames / max(time.time() - t0, 1e-6)
        cv2.putText(out, "FPS %.1f | sockets: %d" % (fps, n), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("socket detector", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            os.makedirs(OUT_DIR, exist_ok=True)
            p = os.path.join(OUT_DIR, "snap_%d.jpg" % int(time.time()))
            cv2.imwrite(p, out)
            print("已截图:", p)
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main():
    ap = argparse.ArgumentParser(description="插座检测程序")
    ap.add_argument("source", nargs="*", help="图片 / 视频 / 文件夹 / 0(摄像头)")
    ap.add_argument("--weights", default=None, help="权重，默认自动找最新 best.pt")
    ap.add_argument("--conf", type=float, default=0.35, help="置信度阈值 (默认 0.35)")
    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸 (默认 640)")
    ap.add_argument("--no-open", action="store_true", help="处理完不自动打开结果")
    ap.add_argument("--show", action="store_true", help="处理视频时同步显示画面")
    args = ap.parse_args()

    if not args.source:
        print("请给出要检测的东西，例如：")
        print("    python detect.py 图片.jpg")
        print("    python detect.py 文件夹")
        print("    python detect.py 0          (摄像头)")
        print("或者直接把文件拖到 拖到这里检测.bat 上。")
        return 1

    weights = args.weights or find_latest_best()
    if not weights or not os.path.exists(weights):
        print("找不到权重: %s" % weights)
        print("请先运行 python train.py 训练，或用 --weights 指定 .pt 文件。")
        return 1

    print("=" * 60)
    print("权重:", weights)
    model = YOLO(weights)
    print("模型已加载，类别:", model.names)
    print("置信度阈值:", args.conf)
    print("结果目录:", OUT_DIR)
    print("=" * 60)

    for src in args.source:
        if str(src).isdigit():
            return do_camera(model, args.conf, args.imgsz)

        if not os.path.exists(src):
            print("找不到:", src)
            continue

        if os.path.isdir(src):
            files = sorted(
                f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT
            )
            if not files:
                print("文件夹里没有图片:", src)
                continue
            print("[文件夹] %s 共 %d 张" % (src, len(files)))
            for f in files:
                do_image(model, os.path.join(src, f), args.conf, args.imgsz, False)
            if not args.no_open:
                open_path(OUT_DIR)
        else:
            ext = os.path.splitext(src)[1].lower()
            print("[文件]", src)
            if ext in IMG_EXT:
                do_image(model, src, args.conf, args.imgsz, not args.no_open)
            else:
                do_video(model, src, args.conf, args.imgsz,
                         not args.no_open, args.show)

    return 0


if __name__ == "__main__":
    code = main()
    if code:
        print("退出码:", code)
