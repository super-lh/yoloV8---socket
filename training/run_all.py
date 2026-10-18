# -*- coding: utf-8 -*-
"""
一键串起后续流程：检查标注 → 生成预览 → 训练 → 推理

标注（第 1 步，需人工）：
    python start_labeling.py              # 会弹出 labelImg 窗口，标完关掉

本脚本自动跑完剩下的部分（不用一步步敲命令）：
    python run_all.py                     # 检查 → 预览 → 训练 → 推理
    python run_all.py --skip-train        # 只检查 + 预览 + 用已有权重推理
    python run_all.py --epochs 200        # 改训练轮数
    python run_all.py --source 图片.jpg    # 改推理输入

说明：所有子脚本都会自动把工作目录切到 D:\\YOLOv8\\training，
      所以你在哪个文件夹打开 VS Code、按 F5 都能跑。
"""
import argparse
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

# 【关键】工作目录固定到脚本所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

STEPS = [
    ("1/4 检查标注", ["check_dataset.py"]),
    ("2/4 生成标注预览", ["preview_labels.py"]),
    ("3/4 训练", ["train.py", "--strict"]),
    ("4/4 推理", ["predict.py"]),
]


def run_step(title, script_args, extra=()):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)
    cmd = [sys.executable] + list(script_args) + list(extra)
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print()
        print("!! 这一步失败（退出码 %d），后续步骤已跳过。" % ret)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="一键跑通 检查→预览→训练→推理")
    ap.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")
    ap.add_argument("--batch", type=int, default=8, help="批大小 (默认: 8)")
    ap.add_argument("--name", default="socket_v2", help="本次训练名称 (默认: socket_v2)")
    ap.add_argument("--source", default="../dataset/images/val/socket_03.jpg", help="推理用的图片/文件夹")
    ap.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值 (默认: 0.25)")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练，直接用现有最新权重推理")
    args = ap.parse_args()

    print("工作目录:", os.getcwd())
    print("计划步骤:", " → ".join(t for t, _ in STEPS))

    if not run_step(STEPS[0][0], STEPS[0][1]):
        return 1
    if not run_step(STEPS[1][0], STEPS[1][1]):
        return 1

    if not args.skip_train:
        if not run_step(STEPS[2][0], STEPS[2][1], ["--epochs", str(args.epochs),
                                                   "--batch", str(args.batch),
                                                   "--name", args.name]):
            return 1

    if not run_step(STEPS[3][0], STEPS[3][1], ["--source", args.source, "--conf", str(args.conf)]):
        return 1

    print()
    print("=" * 62)
    print("全流程结束")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    code = main()
    if code and "--strict" in sys.argv:
        raise SystemExit(code)
