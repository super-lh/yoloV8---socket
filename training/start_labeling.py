# -*- coding: utf-8 -*-
"""
一键启动 labelImg 进行标注

用法：
    python start_labeling.py              # 标注训练集（默认）
    python start_labeling.py --split val  # 标注验证集

重要：必须用 labelImg.exe 启动。
      "python -m labelImg" 方式无效 —— 该包的 __init__.py 是空的，
      命令会立刻结束且没有任何报错，看上去就像"没打开"。

启动后请在 labelImg 里做两件事（很关键）：
  1. 左侧 "PascalVOC" 按钮点一下，切换成 "YOLO"（否则导出的是 xml，训练读不了）
  2. 菜单 File → Change Save Dir，确认保存目录是 labels\\train（或 labels\\val）

标注操作：
  w 键 = 画框        Ctrl+S = 保存        d 键 = 下一张        a 键 = 上一张
  每张图保存后会生成一个同名 .txt 文件

标完后：
  python check_dataset.py                # 体检一遍标注
  python train.py                        # 开始训练
"""
import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple

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

BASE = r"D:\YOLOv8\dataset"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_launcher() -> Tuple[Optional[List[str]], Optional[str]]:
    """找出能真正启动 labelImg 的方式，返回 (启动命令, 错误信息)"""
    cands: List[List[str]] = []

    # 1) Scripts 目录下的可执行文件（最可靠）
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    for exe in ("labelImg.exe", "labelImg"):
        p = os.path.join(scripts_dir, exe)
        if os.path.exists(p):
            cands.append([p])

    # 2) 找到 labelImg 的源码主文件，直接用解释器跑
    try:
        import labelImg as _li  # type: ignore

        pkg_dir = getattr(_li, "__path__", None)
        if pkg_dir:
            main_py = os.path.join(list(pkg_dir)[0], "labelImg.py")
            if os.path.exists(main_py):
                cands.append([sys.executable, main_py])
    except ImportError:
        pass

    if not cands:
        return None, "找不到 labelImg 启动入口，请先在 base 环境执行：pip install labelImg"
    return cands[0], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"], help="标注哪个集合")
    ap.add_argument(
        "--img-dir",
        default=None,
        help="临时指定图片目录（一般不用，走默认的数据集目录）",
    )
    args = ap.parse_args()

    img_dir = args.img_dir or os.path.join(BASE, "images", args.split)
    label_dir = os.path.join(BASE, "labels", args.split)
    classes = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.txt")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    n_img = len([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXT])

    launcher, err = find_launcher()
    if err is not None or launcher is None:
        print(err or "找不到可用的 labelImg 启动入口")
        sys.exit(1)

    with open(classes, encoding="utf-8") as fh:
        class_names = fh.read().strip()

    print("=" * 60)
    print("图片目录 :", img_dir, "(当前 %d 张)" % n_img)
    print("标注保存 :", label_dir)
    print("类别文件 :", classes, "->", class_names)
    print("启动方式 :", " ".join(launcher))
    print("=" * 60)
    if n_img == 0:
        print("!! 目录里还没有图片。")
        print("   可以把照片先放进来再启动；或者在 labelImg 里 File → Open Dir 临时选别的目录。")
    print()
    print("正在启动 labelImg，窗口会稍后出现（首次加载约 2-5 秒）...")

    # labelImg 的用法: labelImg [图片目录或图片] [预定义类别文件]
    cmd = list(launcher) + [img_dir, classes]

    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print()
        print("!! labelImg 退出码 %d，若窗口没出现，把上面的报错发给我。" % ret)
    else:
        print()
        print("labelImg 已关闭。下一步：")
        print("    python check_dataset.py")


if __name__ == "__main__":
    main()
