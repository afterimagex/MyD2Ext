#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
import sys

import onnx
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup

sys.path.append('..')
from MyD2Ext.projects.retinaface.config import add_retinaface_config
from onnxsim import simplify


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


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    with torch.no_grad():
        inputs = torch.randn(1, 3, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST).to(model.device)
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                export_params=True,
                verbose=False,
                # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                do_constant_folding=True,
                input_names=['input'],
                # output_names=['prob', 'bbox', 'keypoint'],
                opset_version=10,
                # custom_opsets=10,
                # keep_initializers_as_inputs=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    # all_passes = onnx.optimizer.get_available_passes()
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    # assert all(p in all_passes for p in passes)
    # onnx_model = onnx.optimizer.optimize(onnx_model, passes)

    model_simp, check = simplify(onnx_model)
    model_simp = remove_initializer_from_input(model_simp)
    assert check, "Simplified ONNX model could not be validated"

    save_onnx_name = cfg.MODEL.WEIGHTS.replace('.pth', '.onnx')
    onnx.save_model(model_simp, save_onnx_name)
    print(f"Export onnx model in {save_onnx_name} successfully!")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
