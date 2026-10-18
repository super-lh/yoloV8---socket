# -*- coding: utf-8 -*-
"""
YOLOv8 (ultralytics 8.0.5) 推理示例

注意：8.0.5 是旧版 API，与新版 (8.4.x) 用法差异较大：
  - predict() 默认不返回结果（返回 None），需传 return_outputs=True
  - return_outputs=True 时返回生成器，元素为 dict，检测框在 r['det'] 中
  - 检测框格式：[x1, y1, x2, y2, conf, cls]
"""
from ultralytics import YOLO

WEIGHTS = r"D:\YOLOv8\weights\yolov8n.pt"
SOURCE = "ultralytics/assets/bus.jpg"
PROJECT = r"D:\YOLOv8\output"

model = YOLO(WEIGHTS)
print("模型加载完成 | 类别数:", len(model.model.names))

# 方式一：标准推理，结果图保存到磁盘，检测信息打印在控制台
print("\n[方式一] 推理并保存结果图")
model.predict(source=SOURCE, device=0, save=True, project=PROJECT, name="demo", exist_ok=True)

# 方式二：编程方式获取检测框（8.0.5 专用写法）
print("\n[方式二] 逐条读取检测框")
names = model.model.names
for r in model.predict(source=SOURCE, device=0, return_outputs=True):
    for *xyxy, conf, cls in r["det"].tolist():
        box = [round(v, 1) for v in xyxy]
        print("  %-12s 置信度 %.2f  坐标 %s" % (names[int(cls)], conf, box))

print("\n结果图目录:", PROJECT + r"\demo")
