"""Calibrate the anomaly threshold from normal (good) images only.

This is for the "train on normal parts, then output y/n" workflow: it runs the
trained model over a folder of known-normal images, takes the maximum anomaly
score, adds a safety margin, and writes the result back into config.yml as the
algorithm-specific threshold.

Usage:
    python scripts/calibrate_threshold.py --config config.yml
                                           [--img_path dataset/bottle/train/good]
                                           [--margin 0.05]
"""

import argparse
import os

import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import anomavision
from anomavision.config import load_config
from anomavision.inference.model.wrapper import ModelWrapper
from anomavision.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate threshold from normal images")
    parser.add_argument("--config", required=True, help="Path to config.yml")
    parser.add_argument("--img_path", default=None, help="Folder of known-normal images (default: config.img_path)")
    parser.add_argument("--margin", type=float, default=0.05, help="Safety margin added to max score")
    args = parser.parse_args()

    setup_logging(enabled=True, log_level="INFO", log_to_file=False)
    cfg = edict(load_config(args.config))

    normal_dir = args.img_path or cfg.get("img_path")
    if not normal_dir or not os.path.isdir(normal_dir):
        raise SystemExit(f"Normal-image folder not found: {normal_dir}")

    model_path = os.path.join(
        cfg.model_data_path, cfg.algorithm, cfg.class_name, cfg.run_name, cfg.model
    )
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    model = ModelWrapper(model_path, "cpu")
    ds = anomavision.AnodetDataset(
        normal_dir,
        resize=(cfg.resize[0], cfg.resize[1]),
        crop_size=cfg.get("crop_size") or None,
        normalize=cfg.get("normalize", True),
        mean=cfg.get("norm_mean"),
        std=cfg.get("norm_std"),
    )
    dl = DataLoader(ds, batch_size=1, num_workers=0)

    scores = []
    for batch, _, _, _ in dl:
        s, _ = model.predict(batch)
        scores.extend(np.asarray(s).reshape(-1).tolist())
    model.close()

    scores = np.array(scores)
    max_score = float(scores.max())
    threshold = round(max_score + args.margin, 4)
    print(f"Normal images evaluated: {len(scores)}")
    print(f"Score min/mean/max: {scores.min():.4f} / {scores.mean():.4f} / {max_score:.4f}")
    print(f"Recommended threshold (max + margin {args.margin}): {threshold}")

    key = f"thresh_{str(cfg.algorithm).lower()}"
    old = cfg.get(key)
    cfg[key] = threshold
    with open(args.config, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(key + ":"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{key}: {threshold}\n"
            break
    with open(args.config, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Updated {args.config}: {key} {old} -> {threshold}")


if __name__ == "__main__":
    main()
