# -*- coding: utf-8 -*-
"""查询 Zenodo 插座数据集的文件列表与下载链接"""
import json
import urllib.request

URL = "https://zenodo.org/api/records/18835198"

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
except Exception as e:
    print("请求失败:", e)
    raise SystemExit(1)

print("标题:", data.get("title"))
print("DOI :", data.get("doi"))
print()

files = data.get("files", [])
print("文件数:", len(files))
for f in files:
    size = f.get("size", 0)
    print("-" * 60)
    print("文件名:", f.get("key"))
    print("大小  : %.1f MB" % (size / 1024 / 1024))
    link = f.get("links", {}).get("self") or f.get("links", {}).get("download")
    print("下载  :", link)
