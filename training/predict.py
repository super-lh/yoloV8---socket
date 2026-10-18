# -*- coding: utf-8 -*-
"""
YOLOv8 推理脚本

用法：
    python predict.py                                    # 自动找最新训练好的 best.pt，配合 --source 用
    python predict.py --source test.jpg
    python predict.py --source 图片文件夹
    python predict.py --source video.mp4
    python predict.py --source 0                         # 摄像头

    # 指定权重
    python predict.py --weights runs/detect/runs_train/socket_v1/weights/best.pt --source test.jpg

    # 还没训练好？用官方预训练权重跑通用检测（80 类，不含插座）
    python predict.py --weights ../weights/yolov8n.pt --source ../output/demo/bus.jpg

权重默认值：留空时自动在 runs/ 下找 epoch 最新的 best.pt，找不到再退回 yolov8n.pt。
说明：用 getattr 读结果属性是为了避开 Pylance 对 ultralytics 返回类型的错误推断。
"""
import argparse
import glob
import os
import sys

# 让终端能以 UTF-8 打印中文路径（只在支持 reconfigure 的新版解释器上生效）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# 【关键】把工作目录固定到脚本所在目录 D:\YOLOv8\training
# 这样无论从哪个文件夹按 F5，runs/ 和相对路径都能正确解析
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)


def find_latest_best():
    """在 runs/ 下找最近训练出来的 best.pt（按修改时间倒序）"""
    cands = []
    for pat in ("runs/**/best.pt", "runs/**/last.pt", "runs_train/**/best.pt"):
        cands += glob.glob(pat, recursive=True)
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return os.path.normpath(cands[0])


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 推理脚本")
    ap.add_argument(
        "--weights",
        default=None,
        help="权重路径；不填则自动取 runs/ 下最新的 best.pt",
    )
    ap.add_argument("--source", default="test.jpg", help="图片 / 文件夹 / 视频 / 摄像头编号")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (默认: 0.5)")
    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸 (默认: 640)")
    ap.add_argument("--device", default=None, help="GPU 编号，CPU 填 cpu (默认: 自动)")
    ap.add_argument("--project", default="runs_predict", help="输出目录 (默认: runs_predict)")
    ap.add_argument("--name", default="exp", help="输出子目录名 (默认: exp)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="出错时返回非零退出码（供脚本串联）；不加则始终返回 0，调试时不会中断",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # 1) 决定用哪个权重
    if args.weights:
        weights = args.weights
    else:
        weights = find_latest_best()
        if not weights:
            for fallback in ("yolov8n.pt", os.path.join("..", "weights", "yolov8n.pt")):
                if os.path.exists(fallback):
                    weights = fallback
                    break
        if not weights:
            print("找不到任何权重文件。")
            print("当前工作目录: %s" % os.getcwd())
            print("请先训练（python train.py），或用 --weights 指定 .pt 路径。")
            return 1
        print("自动选用权重:", weights)

    if not os.path.exists(weights):
        print("找不到权重文件: %s" % weights)
        print("当前工作目录: %s" % os.getcwd())
        print()
        print("可用的权重：")
        found = []
        for pat in ("*.pt", os.path.join("weights", "*.pt"), os.path.join("..", "weights", "*.pt"),
                    "runs/**/best.pt", "runs/**/last.pt"):
            found += glob.glob(pat, recursive=True)
        for f in sorted({os.path.normpath(x) for x in found}):
            print("   ", f)
        print()
        print("提示：不填 --weights 时会自动挑 runs/ 下最新的 best.pt。")
        return 1

    if not os.path.exists(args.source):
        print("找不到推理来源: %s" % args.source)
        print("当前工作目录: %s" % os.getcwd())
        print("请用 --source 指定图片/文件夹/视频，或填 0 用摄像头。")
        return 1

    # 2) 决定设备
    if args.device:
        device = args.device
    else:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("权重:", weights)
    print("来源:", args.source, "| 置信度阈值:", args.conf, "| 设备:", device)

    model = YOLO(weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    total = 0
    save_dir = None
    for r in results:
        names = getattr(r, "names", {}) or {}
        boxes = getattr(r, "boxes", None)
        path = str(getattr(r, "path", "image"))
        save_dir = getattr(r, "save_dir", save_dir)
        n = len(boxes) if boxes is not None else 0
        total += n
        print()
        print("[%s] 检出 %d 个目标" % (os.path.basename(path), n))
        if boxes is None:
            continue
        for i in range(n):
            cls_id = int(boxes.cls[i])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(boxes.conf[i])
            xyxy = [round(float(v), 1) for v in boxes.xyxy[i].tolist()]
            print("   %-15s %.2f  坐标 %s" % (cls_name, conf, xyxy))

    print()
    print("共检出 %d 个目标" % total)
    print("结果已保存到:", save_dir or os.path.join(args.project, args.name))
    return 0


if __name__ == "__main__":
    # 默认丢弃返回值 → 进程永远 exit 0，VS Code 调试器不会把 SystemExit 当异常弹框。
    # 需要退出码供脚本串联时加 --strict（需放在参数最前面或用 --strict 显式传入）。
    argv = sys.argv[1:]
    strict = "--strict" in argv
    if strict:
        argv = [a for a in argv if a != "--strict"]
        sys.argv = [sys.argv[0]] + argv
    code = main()
    if code and strict:
        raise SystemExit(code)
