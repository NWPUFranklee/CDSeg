#!/usr/bin/env python3
"""
Simple script to generate CAM visualizations for CAT-Seg using the existing
`cat_seg/cam.py` helper and the model implementation in this repo.

Usage example:
python tools/generate_cams.py --config-file configs/vitb_384.yaml --weights output/model_0059999.pth --classes 0,5 --outdir output/cams

This script performs minimal repo edits: it does NOT modify training code. It
loads the model, runs a single forward on the first test sample (or an input
image if provided), registers hooks on the internal Aggregator, backprops from
raw logits (the model must save `_last_raw_outputs` in forward), and saves
visualizations to disk.
"""
import argparse
import os
import torch
import numpy as np
import sys

# Ensure repo root is on sys.path so `import cat_seg` works when running this script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog

# repo helpers
from cat_seg.cam import MultiPositionGradCAM
from cat_seg import MaskFormerSemanticDatasetMapper, MaskFormerPanopticDatasetMapper, DETRPanopticDatasetMapper

# reuse the project config extensions
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg.config import add_cat_seg_config


def build_cfg(config_file: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def load_model_from_cfg(cfg, weights: str):
    # build model via the project's Trainer (DefaultTrainer subclass)
    # import here to avoid circular effects at module import time
    from train_net import Trainer
    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(weights, resume=False)
    return model


def get_mapper(cfg):
    name = cfg.INPUT.DATASET_MAPPER_NAME
    if name == "mask_former_semantic":
        return MaskFormerSemanticDatasetMapper(cfg, False)
    elif name == "mask_former_panoptic":
        return MaskFormerPanopticDatasetMapper(cfg, False)
    elif name == "detr_panoptic":
        return DETRPanopticDatasetMapper(cfg, False)
    else:
        return MaskFormerSemanticDatasetMapper(cfg, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", default="0", help="Comma-separated class indices, e.g. '0,5'")
    parser.add_argument("--outdir", default=None, help="Where to save CAM images (defaults to cfg.OUTPUT_DIR/cams)")
    parser.add_argument("--samples", type=int, default=1, help="Number of dataset samples to run (default 1)")
    args = parser.parse_args()

    cfg = build_cfg(args.config_file)
    outdir = args.outdir or os.path.join(cfg.OUTPUT_DIR, "cams")
    os.makedirs(outdir, exist_ok=True)

    print("Building model...")
    model = load_model_from_cfg(cfg, args.weights)
    device = next(model.parameters()).device
    model.to(device)
    model.eval()

    # dataset
    if len(cfg.DATASETS.TEST) == 0:
        raise RuntimeError("No TEST dataset configured in cfg.DATASETS.TEST")
    dataset_name = cfg.DATASETS.TEST[0]
    dataset = DatasetCatalog.get(dataset_name)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {dataset_name} is empty")

    mapper = get_mapper(cfg)

    # prepare samples
    samples = dataset[: args.samples]

    gradcam = MultiPositionGradCAM(model)

    class_list = [int(x) for x in args.classes.split(",") if x.strip() != ""]

    for si, sample in enumerate(samples):
        mapped = mapper(sample)
        # move tensors to device
        if "image" in mapped:
            mapped["image"] = mapped["image"].to(device)
        batched = [mapped]

        for cls in class_list:
            print(f"Running sample {si} class {cls}...")
            try:
                cams = gradcam.generate_from_batched_forward(model, batched, int(cls))
            except Exception as e:
                print("Failed to generate CAM:", e)
                continue

            orig = getattr(model, "_last_input_image", None)
            if orig is None:
                print("No original image found on model; skipping save")
                continue

            save_path = os.path.join(outdir, f"sample{si}_class{cls}.png")
            try:
                gradcam.visualize_multi_position_cam(orig, cams, class_name=str(cls), save_path=save_path)
                print("Saved:", save_path)
            except Exception as e:
                print("Failed to save visualization:", e)

    print("Done.")


if __name__ == "__main__":
    main()
