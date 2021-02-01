#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup

sys.path.append('../..')
sys.path.append('..')

from d2ext.config.defaults import add_retinaface_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_retinaface_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.META_ARCHITECTURE = '_RetinaFace'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    save_onnx_name = cfg.MODEL.WEIGHTS.replace('.pth', '.onnx')
    model.eval()
    # inputs = torch.randn(1, 3, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST).to(model.device)
    inputs = torch.randn(1, 3, 640, 640).to(model.device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            save_onnx_name,
            export_params=True,
            verbose=False,
            training=False,
            do_constant_folding=True,
            keep_initializers_as_inputs=True,
            # input_names=['input'],
            # output_names=['output'],
            # opset_version=9,
        )

        # with io.BytesIO() as f:
        #     torch.onnx.export(
        #         model,
        #         inputs,
        #         f,
        #         export_params=True,
        #         verbose=False,
        #         # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        #         # do_constant_folding=True,
        #         # input_names=['input'],
        #         # output_names=['prob', 'bbox', 'keypoint'],
        #         # opset_version=10,
        #         # custom_opsets=10,
        #         # keep_initializers_as_inputs=True,
        #     )
        #     onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    # all_passes = onnx.optimizer.get_available_passes()
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    # assert all(p in all_passes for p in passes)
    # onnx_model = onnx.optimizer.optimize(onnx_model, passes)

    # model_simp, check = simplify(onnx_model)
    # model_simp = remove_initializer_from_input(model_simp)
    # assert check, "Simplified ONNX model could not be validated"
    #
    # onnx.save_model(model_simp, save_onnx_name)
    print(f"Export onnx model in {save_onnx_name} successfully!")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
