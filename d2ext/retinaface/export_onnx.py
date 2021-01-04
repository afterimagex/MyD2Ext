#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup

sys.path.append('..')

from retinaface.config import add_retinaface_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_retinaface_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.META_ARCHITECTURE = 'RetinaFaceDeploy'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(1, 3, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST)
        torch.onnx.export(
            model,
            inputs,
            cfg.MODEL.WEIGHTS.replace('.pth', '.onnx'),
            export_params=True,
            verbose=False,
            input_names=['input:0'],
            output_names=['output:0'],
            # opset_version=9,
            # custom_opsets=10,
        )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
