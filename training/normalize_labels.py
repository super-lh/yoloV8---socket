# -*- coding: utf-8 -*-
"""
把标注整理成 YOLOv8 要求的格式

本数据集的实际情况：标注已经是 YOLO 的 "class cx cy w h"（归一化），
唯一的问题是混入了 class 1（整面墙/整图的大框），而 data.yaml 只定义了 1 类 socket。
本脚本按你的选择，只保留 class 0(socket)，其余行删除；
原本只有 class 1 的文件会被写成空 txt（= 负样本，YOLOv8 支持且需要）。

用法：
    python normalize_labels.py            # 先预览会怎么改（不写盘）
    python normalize_labels.py --apply    # 真正改写标注文件

原始标注已备份在 D:\\YOLOv8\\dataset\\labels_原始备份
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 【关键】把工作目录固定到脚本所在目录 D:\YOLOv8\training
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

BASE = r"D:\YOLOv8\dataset"


def parse_line(ln):
    """返回 (cls, cx, cy, bw, bh) 或 None（格式不合法）"""
    parts = ln.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:5]]
    except ValueError:
        return None
    if min(vals) < 0 or max(vals) > 1:
        return None
    return (cls, vals[0], vals[1], vals[2], vals[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写盘；不加则只预览")
    ap.add_argument("--keep-class", type=int, default=0, help="保留哪个类别 id (默认: 0)")
    args = ap.parse_args()

    total_before = total_after = 0
    changed_files = 0
    emptied = []
    bad_lines = []

    for split in ("train", "val"):
        ldir = os.path.join(BASE, "labels", split)
        idir = os.path.join(BASE, "images", split)
        if not os.path.isdir(ldir):
            continue
        print("=" * 60)
        print("[%s]" % split)
        for fn in sorted(os.listdir(ldir)):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(ldir, fn)
            raw = open(path, encoding="utf-8", errors="replace").read().splitlines()
            keep, drop, bad = [], 0, 0
            for i, ln in enumerate(raw, 1):
                if not ln.strip():
                    continue
                p = parse_line(ln)
                if p is None:
                    bad += 1
                    bad_lines.append("%s/%s:%d -> %r" % (split, fn, i, ln[:60]))
                    continue
                if p[0] == args.keep_class:
                    keep.append(p)
                else:
                    drop += 1
            total_before += len(keep) + drop
            total_after += len(keep)

            new_txt = "".join(
                "%d %.6f %.6f %.6f %.6f\n" % (p[0], p[1], p[2], p[3], p[4]) for p in keep
            )
            old_txt = open(path, encoding="utf-8", errors="replace").read()
            if new_txt != old_txt:
                changed_files += 1
                if not keep:
                    emptied.append("%s/%s" % (split, fn))
            if drop or bad:
                img_ok = os.path.exists(os.path.join(idir, fn.replace(".txt", ".jpg")))
                print(
                    "  %-18s 保留 %d 个框, 删除 %d 个 class!=%d 的框%s, 非法行 %d%s"
                    % (fn, len(keep), drop, args.keep_class, "" if img_ok else " (!!无对应图片)", bad, "")
                )
            if args.apply:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new_txt)

    print("=" * 60)
    print("框总数: %d -> %d (删除 %d 个)" % (total_before, total_after, total_before - total_after))
    print("受影响文件数: %d" % changed_files)
    if emptied:
        print("变成空文件(负样本)的: %s" % ", ".join(emptied))
    if bad_lines:
        print("!! 非法行（已跳过）:")
        for b in bad_lines[:10]:
            print("   ", b)

    if not args.apply:
        print()
        print("这是预览，未写盘。确认后执行：python normalize_labels.py --apply")
    else:
        print()
        print("已写盘。下一步：python check_dataset.py")


if __name__ == "__main__":
    main()
