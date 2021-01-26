# ANNETTE - Accurate Neural Network Execution Time Estimation

# Install
* I recommended install in a virtualenv
* Tested with python 3.6-3.7
`pip3 install -e .`

# Usage
* mmtoir to export to MMDNN format
* ANNETTE only needs the .pb-file, you can savely remove the .npy-weightfile

## MMDNN to ANNETTE
* use `python3 src/annette/m2a.py [network-file/network-name]` to convert from mmdnn to annette format. Output is stored in `database/graphs/annette/[network].json`
* Either copy the file to `database/graphs/mmdnn/[network].pb` and use only the `network-name` oder give the full Path

## Estimation
`python3 src/annette/estimate.py [network-name] [mapping-model] [layer-model]`

## Examples
mmtoir -f tensorflow -d database/graphs/mmdnn/tf_deeplabv3_original --inputShape 1025,2049,3 database/graphs/tf/deeplabv3_original.pb

### DeeplabV3 (https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
```
mmtoir -f tensorflow -w database/graphs/tf/deeplabv3_mnv2_dm05_pascal.pb --inNodeName MobilenetV2/MobilenetV2/input --inputShape 513,513,3 --dstNodeName ArgMax -o database/graphs/mmdnn/deeplabv3
python3 src/annette/m2a.py deeplabv3 
python3 src/annette/estimate.py deeplabv3 ov2 ncs2-mixed
```

### Mobilenetv1
```
mmdownload -f tensorflow -n mobilenet_v1_1.0_frozen -o database/graphs/tf/
mmtoir -f tensorflow -w database/graphs/tf/mobilenet_v1_1.0_224/frozen_graph.pb --inNodeName input --inputShape 224,224,3 --dstNodeName MobilenetV1/Predictions/Softmax -o database/graphs/mmdnn/mobilenet_v1
python3 src/annette/m2a.py mobilenet_v1
python3 src/annette/estimate.py mobilenet_v1 ov2 ncs2-mixed
```

### TinyYolov4
```
mmtoir -f tensorflow -w database/graphs/tf/darknet_yolov4_tiny_512_512_coco.pb --inNodeName inputs --inputShape 512,512,3 --dstNodeName detector/yolo-v4-tiny/Conv_17/BiasAdd detector/yolo-v4-tiny/Conv_20/BiasAdd -o database/graphs/mmdnn/darknet_yolov4_tiny_512_512_coco
python3 src/annette/m2a.py darknet_yolov4_tiny_512_512_coco
python3 src/annette/estimate.py darknet_yolov4_tiny_512_512_coco ov2 ncs2-mixed
```

### Yolov4
```
mmtoir -f tensorflow -w database/graphs/tf/darknet_yolov4_512_512_coco.pb --inNodeName inputs --inputShape 512,512,3 --dstNodeName detector/yolo-v4/Conv_1/BiasAdd detector/yolo-v4/Conv_9/BiasAdd detector/yolo-v4/Conv_17/BiasAdd -o database/graphs/mmdnn/darknet_yolov4_512_512_coco
python3 src/annette/m2a.py darknet_yolov4_512_512_coco
python3 src/annette/estimate.py darknet_yolov4_512_512_coco ov2 ncs2-mixed
```

### Pytorch Mnasnet0.5
```
mmdownload -f pytorch -n mnasnet0_5 -o database/graphs/pytorch/
mmtoir -f pytorch -d database/graphs/mmdnn/mnasnet0_5 --inputShape 3,224,224 -n database/graphs/pytorch/imagenet_mnasnet0_5.pth
python3 src/annette/m2a.py mnasnet0_5
python3 src/annette/estimate.py mnasnet0_5 ov2 ncs2-roofline
```

### Pytorch Densenet121
```
mmdownload -f pytorch -n densenet121 -o database/graphs/pytorch/
mmtoir -f pytorch -d database/graphs/mmdnn/densenet121 --inputShape 3,224,224 -n database/graphs/pytorch/imagenet_densenet121.pth
python3 src/annette/m2a.py densenet121 
python3 src/annette/estimate.py densenet121 ov2 ncs2-roofline
```

### Pytorch Squeezenet1.0
```
mmdownload -f pytorch -n squeezenet1_0 -o database/graphs/pytorch/
mmtoir -f pytorch -d database/graphs/mmdnn/squeezenet1_0 --inputShape 3,224,224 -n database/graphs/pytorch/imagenet_squeezenet1_0.pth
python3 src/annette/m2a.py squeezenet1_0 
python3 src/annette/estimate.py squeezenet1_0 ov2 ncs2-roofline
```

## Note
This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
