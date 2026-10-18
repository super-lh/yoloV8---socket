# -*- coding: utf-8 -*-
"""
把已标注的训练图片按比例划一部分到验证集

典型流程：
  1. 所有照片放进 dataset/images/train 并全部标注
  2. python split_dataset.py                 # 预览会移动哪些
  3. python split_dataset.py --apply         # 真正执行移动
  4. python check_dataset.py                 # 确认划分无误

用法：
    python split_dataset.py                  # 预览（dry-run，不真动）
    python split_dataset.py --apply          # 真正执行移动
    python split_dataset.py --ratio 0.25     # 改验证集比例
    python split_dataset.py --seed 7         # 换随机种子得到不同划分

注意：本脚本默认不抛 SystemExit，在 VS Code 里按 F5 不会弹「发生异常」红框；
      需要退出码时加 --strict。
"""
import argparse
import os
import random
import shutil
import sys
from typing import List

# 让终端能以 UTF-8 打印中文路径（只在支持 reconfigure 的新版解释器上生效）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

# 【关键】把工作目录固定到脚本所在目录 D:\YOLOv8\training
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

BASE = r"D:\super-lhyolov8socket\yoloV8---socket\dataset"  # 2026-10-18 修正原路径 D:\YOLOv8\dataset 已不存在
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_imgs(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description="train -> val 数据集划分")
    ap.add_argument("--base", default=BASE, help="数据集根目录")
    ap.add_argument("--ratio", type=float, default=0.2, help="验证集占比 (默认: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    ap.add_argument("--apply", action="store_true", help="真正执行移动；不加则只预览")
    ap.add_argument("--strict", action="store_true", help="出错时返回非零退出码（供脚本串联）")
    args = ap.parse_args()

    img_tr = os.path.join(args.base, "images", "train")
    img_va = os.path.join(args.base, "images", "val")
    lbl_tr = os.path.join(args.base, "labels", "train")
    lbl_va = os.path.join(args.base, "labels", "val")

    imgs = list_imgs(img_tr)
    if not imgs:
        # 先看 train 空但 val 有货的情况：说明数据早就划分过了，不是错误
        n_va = len(list_imgs(img_va))
        print("训练集目录里还没有图片: %s" % img_tr)
        print("验证集当前有 %d 张: %s" % (n_va, img_va))
        print()
        if n_va > 0:
            print("看起来数据已经划分过了，不需要再运行本脚本。")
            print("如果要重新划分，请先把 val 里的图片移回 train。")
        else:
            print("请先把照片放进训练集目录再运行：")
            print("    %s" % img_tr)
        return 1

    # 优先搬运已标注的图
    labeled, unlabeled = [], []
    for im in imgs:
        stem = os.path.splitext(im)[0]
        txt = os.path.join(lbl_tr, stem + ".txt")
        (labeled if os.path.exists(txt) else unlabeled).append(im)

    n_val = max(1, int(round(len(imgs) * args.ratio)))
    random.seed(args.seed)
    pool = labeled[:]
    random.shuffle(pool)
    picked = pool[:n_val]

    n_val_now = len(list_imgs(img_va))
    print("=" * 60)
    print("数据集根目录: %s" % args.base)
    print("训练集图片  : %d  (已标注 %d, 未标注 %d)" % (len(imgs), len(labeled), len(unlabeled)))
    print("验证集现有  : %d 张" % n_val_now)
    print("本次目标    : 再抽 %d 张到 val (占比 %.0f%%, 种子 %d)" % (n_val, args.ratio * 100, args.seed))
    if n_val_now:
        print()
        print("!! 注意：val 里已经有 %d 张了。本脚本只从 train 往 val 搬，" % n_val_now)
        print("   train 里剩下的 %d 张若按 %.0f%% 再抽，会把 val 比例推到 %.0f%%。"
              % (len(imgs), args.ratio * 100, (n_val_now + n_val) * 100.0 / (len(imgs) + n_val_now)))
        print("   如果数据早就划分好了，不要再跑本脚本 —— 直接 python check_dataset.py 即可。")
    if unlabeled:
        print("!! 有 %d 张图还没标注，先把它们标完再划分更合理" % len(unlabeled))
    print("=" * 60)

    if not picked:
        print("没有可搬运的已标注图片（缺 .txt），先标注再划分。")
        return 1

    print("将移动到 val 的图片:")
    for im in picked:
        print("   ", im)
    print("=" * 60)

    if not args.apply:
        print("这是预览模式，未做任何改动。确认无误后加 --apply 执行：")
        print("    python split_dataset.py --apply")
        return 0

    os.makedirs(img_va, exist_ok=True)
    os.makedirs(lbl_va, exist_ok=True)

    moved = 0
    for im in picked:
        stem = os.path.splitext(im)[0]
        shutil.move(os.path.join(img_tr, im), os.path.join(img_va, im))
        src = os.path.join(lbl_tr, stem + ".txt")
        if os.path.exists(src):
            shutil.move(src, os.path.join(lbl_va, stem + ".txt"))
        moved += 1

    print()
    print("已移动 %d 张到验证集" % moved)
    print("现在 train=%d 张, val=%d 张" % (len(list_imgs(img_tr)), len(list_imgs(img_va))))
    print("下一步: python check_dataset.py")
    return 0


if __name__ == "__main__":
    # 默认丢弃返回值，脚本永远以 0 退出 —— 避免 VS Code 调试器把 SystemExit
    # 当「未捕获异常」弹红框。需要退出码时加 --strict。
    code = main()
    if code and "--strict" in sys.argv:
        raise SystemExit(code)
