"""传统组的运行入口。计时口径和结果落盘都在这里统一, B 和 C 不用各写一遍。

    python -m src.traditional.run_traditional --mode descriptor --descriptor lbp --clf linear_svm
    python -m src.traditional.run_traditional --mode bovw --k 512 --clf random_forest
"""

import argparse
import time

from ..metrics import save_result
from . import bovw, features, io


def run_descriptor(args):
    """传统_B 的实验: 单一手工描述子 + 分类器。"""
    df = io.load_manifest(args.manifest)
    names = io.get_class_names(df)

    t0 = time.time()
    tr_img, y_tr = io.load_split(df, "train", args.image_root, limit=args.limit)
    X_tr = features.extract_all(tr_img, args.descriptor)
    feat_train_s = time.time() - t0

    t0 = time.time()
    clf = bovw.build_classifier(args.clf, seed=args.seed)
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    t0 = time.time()
    te_img, y_te = io.load_split(df, "test", args.image_root, limit=args.limit)
    X_te = features.extract_all(te_img, args.descriptor)
    y_score = bovw.scores_of(clf, X_te)
    test_s = time.time() - t0

    return (y_te, y_score.argmax(1), y_score, names,
            {"train_s": feat_train_s + fit_s, "test_s": test_s})


def run_bovw(args):
    """传统_C 的实验: SIFT + BoVW + 分类器。"""
    df = io.load_manifest(args.manifest)
    names = io.get_class_names(df)

    tr_img, y_tr = io.load_split(df, "train", args.image_root, limit=args.limit)

    t0 = time.time()
    desc = bovw.sample_train_descriptors(tr_img, per_image=args.per_image, seed=args.seed)
    codebook = bovw.build_codebook(desc, k=args.k, seed=args.seed)   # 码本只用 train
    X_tr = bovw.encode_split(tr_img, codebook, args.k)
    feat_train_s = time.time() - t0

    t0 = time.time()
    clf = bovw.build_classifier(args.clf, seed=args.seed)
    clf.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    t0 = time.time()
    te_img, y_te = io.load_split(df, "test", args.image_root, limit=args.limit)
    X_te = bovw.encode_split(te_img, codebook, args.k)
    y_score = bovw.scores_of(clf, X_te)
    test_s = time.time() - t0

    # train_s = 建码本 + 特征提取 + 分类器 fit, 三段都算。只算 fit 会严重低估。
    return (y_te, y_score.argmax(1), y_score, names,
            {"train_s": feat_train_s + fit_s, "test_s": test_s})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["descriptor", "bovw"], required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--image_root", default="data/images")
    ap.add_argument("--descriptor", default="color_hist")
    ap.add_argument("--clf", default="linear_svm")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--per_image", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="每个 split 只取前 N 张, 调试用")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    y_true, y_pred, y_score, names, timing = (
        run_descriptor(args) if args.mode == "descriptor" else run_bovw(args))

    name = args.name or (f"{args.descriptor}_{args.clf}" if args.mode == "descriptor"
                         else f"bovw_k{args.k}_{args.clf}")
    path, scores = save_result("results", name, y_true, y_pred, y_score, names, timing)
    print(name, scores)
    print("->", path)


if __name__ == "__main__":
    main()
