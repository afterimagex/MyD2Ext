<!--
 * @Copyright (c) Pvening All Rights Reserved.
 * @Author         : peichao.xu
 * @Github         : https://github.com/afterimagex
 * @Date           : 2020-012-24 12:34:30
 * @FilePath       : /MyD2Ext/README.md
 * @Description    :
 -->

# detectron2 algorithm extension

## RetinaFace

## CenterNet

## TODO

- [x] RetinaFace
- [x] CenterNet
- [ ] DBNet
- [ ] LFFD
- [ ] YoloV3
- [ ] SoloV2
- [ ] SparseR-CNN
- [ ] Yolact

- [ ] knowledge distill
- [ ] better image augmentation


# Issus

1、onnx to tensorrt error:

```python
Loading ONNX file from path epoch_89.onnx...
Beginning ONNX file parsing
Parser ONNX model failed.
In node 0 (importModel): INVALID_GRAPH: Assertion failed: tensors.count(input_name)
```

need change export script:

```
with torch.no_grad():
    torch.onnx.export(
        model,
        inputs,
        save_onnx_name,
        export_params=True,
        verbose=False,
        training=False,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,  # ---->  this line is very important
    )
```



# Acknowledgement

[Detectron2](https://github.com/facebookresearch/detectron2)

[CenterX](https://github.com/JDAI-CV/centerX)

[RetinaFace](https://github.com/tkianai/RetinaFace.detectron2)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)