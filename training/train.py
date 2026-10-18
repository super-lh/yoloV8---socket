# -*- coding: utf-8 -*-
"""
YOLOv8 自定义目标检测 - 通用训练脚本

环境：ultralytics 8.4.176 + torch 2.14 (CUDA)
注意：请在 D:\\super-lhyolov8socket\\yoloV8---socket\\training 目录下运行，这样会加载 base 环境的 8.4.176
      （若放在 ultralytics-main 源码目录内运行，则会加载旧版 8.0.5，API 不兼容）

基本用法：
    python train.py
    python train.py --epochs 200 --batch 8 --name mymodel
    python train.py --strict        # 出错时返回非零退出码（供脚本串联）

参数说明见下方 argparse，或在终端执行 python train.py --help

说明：默认不抛 SystemExit，按 F5 调试时不会弹「发生异常」红框；
      有问题会打印原因并正常结束。
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
# 这样无论你从哪个文件夹打开 VS Code、按 F5，data.yaml 和 runs/ 都能正确解析
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)


def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 通用训练脚本")

    ap.add_argument(
        "--data",
        default="data.yaml",
        help="数据集配置文件路径 (默认: data.yaml)",
    )
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="预训练权重: yolov8n/s/m/l/x.pt，n 最快、x 最准 (默认: yolov8n.pt)",
    )
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")
    ap.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (默认: 640)")
    ap.add_argument(
        "--batch",
        type=int,
        default=8,
        help="批大小，8GB 显存建议 8；显存不足改 4 或 2 (默认: 8)",
    )
    ap.add_argument("--device", default="0", help="GPU 编号，CPU 填 cpu (默认: 0)")
    ap.add_argument("--project", default="runs_train", help="输出目录 (默认: runs_train)")
    ap.add_argument("--name", default="custom", help="本次训练名称 (默认: custom)")
    ap.add_argument(
        "--patience",
        type=int,
        default=30,
        help="早停轮数：多少轮无提升后停止 (默认: 30)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="数据加载线程数，Windows 建议 0-2 (默认: 2)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="从上次中断处继续训练",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="出错时返回非零退出码（供脚本串联）；不加则始终返回 0，调试时不会中断",
    )

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # 环境自检
    cuda_ok = torch.cuda.is_available()
    device = args.device if cuda_ok else "cpu"
    print("=" * 60)
    print("PyTorch      :", torch.__version__)
    print("CUDA 可用    :", cuda_ok)
    if cuda_ok:
        print("GPU          :", torch.cuda.get_device_name(0))
        print("显存         : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))
    print("使用设备     :", device)
    print("预训练权重   :", args.model)
    print("数据集配置   :", args.data)
    print("当前工作目录 :", os.getcwd())
    print("训练轮数     :", args.epochs, "| 批大小:", args.batch, "| 尺寸:", args.imgsz)
    print("=" * 60)

    # 数据集配置与预训练权重都按"当前工作目录"解析，找不到时给明确指引而不是抛异常
    if not os.path.exists(args.data):
        print("找不到数据集配置文件: %s" % args.data)
        print("当前工作目录: %s" % os.getcwd())
        here_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")
        if os.path.exists(here_yaml):
            print()
            print("脚本所在目录下有这个文件：")
            print("    %s" % here_yaml)
            print("说明当前工作目录不是 training。请在 D:\\super-lhyolov8socket\\yoloV8---socket\\training 下运行，")
            print("或者先执行: cd /d D:\\super-lhyolov8socket\\yoloV8---socket\\training")
        return 1

    if not os.path.exists(args.model):
        print("找不到预训练权重: %s" % args.model)
        print("当前工作目录: %s" % os.getcwd())
        cand = []
        for pat in ("*.pt", os.path.join("weights", "*.pt"), os.path.join("..", "weights", "*.pt")):
            cand += glob.glob(pat)
        if cand:
            print("可用的权重：")
            for c in sorted({os.path.normpath(x) for x in cand}):
                print("   ", c)
            print("用 --model 指定其中一个，例如: python train.py --model ../weights/yolov8n.pt")
        return 1

    # 加载模型：优先使用预训练权重做迁移学习（比从零训练收敛快得多）
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        save=True,
    )

    # ultralytics 8.4 会把输出放在 runs/detect/<project>/<name> 下，
    # 这里直接取它真实的 save_dir，避免路径写死导致找不到 best.pt
    save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        save_dir = os.path.join(args.project, args.name)
    save_dir = str(save_dir)
    best = os.path.join(save_dir, "weights", "best.pt")

    print()
    print("=" * 60)
    print("训练完成")
    print("输出目录   :", save_dir)
    print("最佳权重   :", best)
    print("训练曲线   :", os.path.join(save_dir, "results.png"))
    print("下一步:")
    print("  python predict.py --source 你的图片或视频     (会自动使用最新训练结果)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # 默认丢弃返回值 → 进程永远 exit 0，VS Code 调试器不会把 SystemExit 当异常弹框。
    # 需要退出码供脚本串联时加 --strict。
    code = main()
    if code and "--strict" in sys.argv:
        raise SystemExit(code)
