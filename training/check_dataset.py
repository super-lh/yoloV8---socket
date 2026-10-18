# -*- coding: utf-8 -*-
"""
训练前的数据集体检工具

用法：
    python check_dataset.py                # 检查 data.yaml 里配置的数据集
    python check_dataset.py --data xxx.yaml
    python check_dataset.py --strict        # 有问题时返回非零退出码（供自动化脚本串联）

注意：不加 --strict 时脚本始终返回 0，这样在 VS Code 里按 F5 不会因为
      SystemExit 弹红框、也不会在结尾中断。要看"是否通过"，看打印的文字即可。

会检查：
  1. train / val 图片数量，以及标注是否成对
  2. 标注文件格式是否正确（class_id x_center y_center w h，归一化 0~1）
  3. 类别 id 是否超出 data.yaml 里 names 的范围
  4. 每个类别的框数量分布、空标注（负样本）情况
  5. train / val 比例是否合理

全部通过才会打印 "可以开始训练"。
"""
import argparse
import os
import sys

# 让终端能以 UTF-8 打印中文路径（只在支持 reconfigure 的新版解释器上生效）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

import yaml  # noqa: E402

# 【关键】把工作目录固定到脚本所在目录 D:\YOLOv8\training
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(_SCRIPT_DIR):
    os.chdir(_SCRIPT_DIR)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_imgs(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)


def check_split(data_root, img_dir, label_dir, n_cls, split):
    """检查一个子集，返回 (是否发现问题, 统计信息)"""
    imgs = list_imgs(img_dir)
    print("-" * 62)
    print("[%s]" % split)
    print("  图片目录: %s" % img_dir)
    print("  标注目录: %s" % label_dir)
    print("  图片数量: %d" % len(imgs))

    problems = False

    if len(imgs) == 0:
        print("  !! 没有找到任何图片，训练会报 FileNotFoundError")
        return True, 0, {}

    # 标注配对检查
    missing = []      # 有图无标注
    empty_lbl = []    # 标注为空文件（负样本，合法但要提示）
    fmt_err = []      # 格式错误
    cls_count = {}
    box_total = 0

    for im in imgs:
        stem = os.path.splitext(im)[0]
        txt = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(txt):
            missing.append(im)
            continue
        size = os.path.getsize(txt)
        if size == 0:
            empty_lbl.append(im)
            continue
        with open(txt, encoding="utf-8", errors="replace") as fh:
            lines = [ln.strip() for ln in fh.readlines()]
        for i, ln in enumerate(lines, 1):
            if not ln:
                continue
            parts = ln.split()
            if len(parts) not in (5, 6):
                fmt_err.append("%s:%d 列数=%d -> %r" % (stem + ".txt", i, len(parts), ln[:60]))
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                fmt_err.append("%s:%d 类别 id 不是整数 -> %r" % (stem + ".txt", i, parts[0]))
                continue
            vals = []
            bad = False
            for v in parts[1:5]:
                try:
                    vals.append(float(v))
                except ValueError:
                    bad = True
                    break
            if bad:
                fmt_err.append("%s:%d 坐标不是数字 -> %r" % (stem + ".txt", i, ln[:60]))
                continue
            if cid < 0 or cid >= n_cls:
                fmt_err.append("%s:%d 类别 id=%d 超出 names 范围(0~%d)" % (stem + ".txt", i, cid, n_cls - 1))
                continue
            if min(vals) < 0 or max(vals) > 1:
                fmt_err.append("%s:%d 坐标超出 0~1 -> %s" % (stem + ".txt", i, vals))
                continue
            cls_count[cid] = cls_count.get(cid, 0) + 1
            box_total += 1

    print("  标注框总数: %d" % box_total)
    if cls_count:
        print("  类别分布  : %s" % ", ".join("%s=%d" % (k, v) for k, v in sorted(cls_count.items())))

    if missing:
        problems = True
        print("  !! %d 张图缺少标注（训练会被当成无目标图，通常意味着漏标）:" % len(missing))
        for m in missing[:10]:
            print("       - %s" % m)
        if len(missing) > 10:
            print("       ... 还有 %d 个" % (len(missing) - 10))

    if empty_lbl:
        print("  ~~ %d 张图的标注文件为空（作为负样本是允许的，但插座场景不该有这么多）" % len(empty_lbl))
        for m in empty_lbl[:5]:
            print("       - %s" % m)

    if fmt_err:
        problems = True
        print("  !! %d 处标注格式错误:" % len(fmt_err))
        for e in fmt_err[:10]:
            print("       - %s" % e)
        if len(fmt_err) > 10:
            print("       ... 还有 %d 处" % (len(fmt_err) - 10))

    # 有标注无图
    if os.path.isdir(label_dir):
        orphan = []
        stems = {os.path.splitext(i)[0] for i in imgs}
        for f in sorted(os.listdir(label_dir)):
            if not f.endswith(".txt"):
                continue
            if os.path.splitext(f)[0] not in stems:
                orphan.append(f)
        if orphan:
            print("  ~~ %d 个多余标注文件(无对应图片)，不会被用于训练" % len(orphan))
            for o in orphan[:5]:
                print("       - %s" % o)

    if not missing and not fmt_err:
        print("  OK")
    return problems, len(imgs), cls_count


def main():
    ap = argparse.ArgumentParser(description="YOLO 数据集体检")
    ap.add_argument("--data", default="data.yaml", help="数据集配置文件 (默认: data.yaml)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="有问题时返回非零退出码（供脚本串联）；不加则始终返回 0，调试时不会中断",
    )
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print("找不到配置文件: %s" % args.data)
        print("当前工作目录: %s" % os.getcwd())
        print("请在 D:\\super-lhyolov8socket\\yoloV8---socket\\training 目录下运行，或用 --data 指定 yaml 路径。")
        return 1

    with open(args.data, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = cfg.get("path", ".")
    names = cfg.get("names") or {}
    if isinstance(names, list):
        n_cls = len(names)
    else:
        n_cls = (max(names.keys()) + 1) if names else 0

    print("=" * 62)
    print("数据集配置: %s" % os.path.abspath(args.data))
    print("根目录    : %s" % root)
    print("类别      : %s (共 %d 类)" % (dict(names) if not isinstance(names, list) else names, n_cls))

    split_stats = {}
    any_problem = False
    for key in ("train", "val"):
        rel = cfg.get(key)
        if not rel:
            print("!! 配置文件里缺少 %s 路径" % key)
            any_problem = True
            split_stats[key] = 0
            continue
        p, n, dist = check_split(
            root,
            os.path.join(root, rel),
            os.path.join(root, rel.replace("images", "labels", 1)),
            n_cls,
            key,
        )
        any_problem = any_problem or p
        split_stats[key] = n

    print("=" * 62)
    print("统计: train=%d 张, val=%d 张" % (split_stats.get("train", 0), split_stats.get("val", 0)))

    ntr, nva = split_stats.get("train", 0), split_stats.get("val", 0)
    if ntr and nva:
        r = nva / float(ntr + nva)
        print("验证集占比: %.1f%%  (推荐 15%%~25%%)" % (r * 100))
        if r < 0.05:
            print("!! 验证集太少，指标不可信，建议至少留出 10%")
            any_problem = True
        elif r > 0.4:
            print("~~ 验证集偏多，训练数据被浪费了")
    if ntr < 30:
        print("~~ 训练集只有 %d 张，够跑通流程，但精度会很低；建议先 200 张起步" % ntr)

    if any_problem:
        print()
        print(">>> 存在问题，请先按上面提示修正后再训练")
        return 1

    print()
    print(">>> 检查通过，可以开始训练：")
    print("    python train.py")
    print("  或按 F5 选择「训练插座模型 (train.py)」")
    return 0


if __name__ == "__main__":
    # 故意把返回值丢掉：这样脚本永远以 0 退出，调试器（F5）不会再因为
    # SystemExit 弹红框，也不会在正常流程末尾中断。
    # 需要机器可读结果时用 --strict：有问题才返回非零退出码（配合终端脚本串联）。
    exit_code = main()
    if exit_code and "--strict" in sys.argv:
        raise SystemExit(exit_code)
