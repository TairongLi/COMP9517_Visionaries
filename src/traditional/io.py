"""传统组共用的读图工具。只依赖 numpy/pandas/PIL, 不需要 torch。
B 和 C 都从这里拿图, 保证两人用的是同一份 split 和同一套 class_id。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

COLS = ["filepath", "class_id", "species_name", "split"]


def load_manifest(path):
    df = pd.read_csv(path)
    missing = set(COLS) - set(df.columns)
    assert not missing, f"manifest 缺列: {sorted(missing)}"
    return df


def get_class_names(df):
    m = df[["class_id", "species_name"]].drop_duplicates().sort_values("class_id")
    ids = m["class_id"].tolist()
    assert ids == list(range(len(ids))), "class_id 必须是连续的 0..C-1"
    return m["species_name"].tolist()


def load_split(df, split, image_root, resize=256, limit=None):
    """返回 (images, labels)。images 是 uint8 RGB ndarray 列表。
    resize 只是把长边压到 resize 以控制内存/耗时, 描述子内部该 resize 的自己再 resize。
    limit 用于本地调试时只取前 N 张。"""
    rows = df[df["split"] == split].reset_index(drop=True)
    if limit:
        rows = rows.iloc[:limit]
    root = Path(image_root)

    images, labels = [], []
    for _, r in rows.iterrows():
        im = Image.open(root / r["filepath"]).convert("RGB")
        if resize:
            im.thumbnail((resize, resize))
        images.append(np.asarray(im, dtype=np.uint8))
        labels.append(int(r["class_id"]))
    return images, np.array(labels, dtype=np.int64)