
s
Placeholder	DataInput"1
_output_shapes
:�����������"&
shape:�����������
|
detector/truedivRealDivPlaceholderdetector/truediv/y"1
_output_shapes
:�����������"
dtype0

�
!detector/yolo-v3-tiny/Conv/Conv2DConvdetector/truediv"1
_output_shapes
:�����������"
pads

    "
kernel_shape
"
strides
"
data_formatNHWC"
shape
�����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
3detector/yolo-v3-tiny/Conv/BatchNorm/FusedBatchNorm	BatchNorm!detector/yolo-v3-tiny/Conv/Conv2D"
dtype0
"1
_output_shapes
:�����������"

bias("
scale("
data_formatNHWC"
epsilon%��'7
�
$detector/yolo-v3-tiny/Conv/LeakyRelu	LeakyRelu3detector/yolo-v3-tiny/Conv/BatchNorm/FusedBatchNorm"1
_output_shapes
:�����������"
shape
�����������"
dtype0

�
#detector/yolo-v3-tiny/pool2/MaxPoolPool$detector/yolo-v3-tiny/Conv/LeakyRelu"
auto_padVALID"
pooling_typeMAX"
dtype0
"1
_output_shapes
:�����������"
pads

        "
kernel_shape
"
strides
"
data_formatNHWC
�
#detector/yolo-v3-tiny/Conv_1/Conv2DConv#detector/yolo-v3-tiny/pool2/MaxPool"
kernel_shape
 "
strides
"
data_formatNHWC"
shape
�����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"1
_output_shapes
:����������� "
pads

    
�
5detector/yolo-v3-tiny/Conv_1/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_1/Conv2D"
dtype0
"1
_output_shapes
:����������� "

bias("
scale("
data_formatNHWC"
epsilon%��'7
�
&detector/yolo-v3-tiny/Conv_1/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_1/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������� "
shape
����������� 
�
%detector/yolo-v3-tiny/pool2_1/MaxPoolPool&detector/yolo-v3-tiny/Conv_1/LeakyRelu"
pooling_typeMAX"
dtype0
"/
_output_shapes
:���������hh "
pads

        "
kernel_shape
"
strides
"
data_formatNHWC"
auto_padVALID
�
#detector/yolo-v3-tiny/Conv_2/Conv2DConv%detector/yolo-v3-tiny/pool2_1/MaxPool"
shape
���������hh "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������hh@"
pads

    "
kernel_shape
 @"
strides
"
data_formatNHWC
�
5detector/yolo-v3-tiny/Conv_2/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_2/Conv2D"
dtype0
"/
_output_shapes
:���������hh@"

bias("
scale("
data_formatNHWC"
epsilon%��'7
�
&detector/yolo-v3-tiny/Conv_2/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_2/BatchNorm/FusedBatchNorm"
dtype0
"/
_output_shapes
:���������hh@"
shape
���������hh@
�
%detector/yolo-v3-tiny/pool2_2/MaxPoolPool&detector/yolo-v3-tiny/Conv_2/LeakyRelu"
auto_padVALID"
pooling_typeMAX"
dtype0
"/
_output_shapes
:���������44@"
pads

        "
kernel_shape
"
strides
"
data_formatNHWC
�
#detector/yolo-v3-tiny/Conv_3/Conv2DConv%detector/yolo-v3-tiny/pool2_2/MaxPool"
paddingSAME"
dtype0
"0
_output_shapes
:���������44�"
pads

    "
kernel_shape	
@�"
strides
"
data_formatNHWC"
shape
���������44@"
auto_pad
SAME_LOWER
�
5detector/yolo-v3-tiny/Conv_3/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_3/Conv2D"
epsilon%��'7"
dtype0
"0
_output_shapes
:���������44�"

bias("
scale("
data_formatNHWC
�
&detector/yolo-v3-tiny/Conv_3/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_3/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������44�"
shape
���������44�
�
%detector/yolo-v3-tiny/pool2_3/MaxPoolPool&detector/yolo-v3-tiny/Conv_3/LeakyRelu"
pooling_typeMAX"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape
"
strides
"
data_formatNHWC"
auto_padVALID
�
#detector/yolo-v3-tiny/Conv_4/Conv2DConv%detector/yolo-v3-tiny/pool2_3/MaxPool"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape

��"
strides
"
data_formatNHWC
�
5detector/yolo-v3-tiny/Conv_4/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_4/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%��'7"
dtype0

�
&detector/yolo-v3-tiny/Conv_4/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_4/BatchNorm/FusedBatchNorm"0
_output_shapes
:����������"
shape
����������"
dtype0

�
%detector/yolo-v3-tiny/pool2_4/MaxPoolPool&detector/yolo-v3-tiny/Conv_4/LeakyRelu"
pooling_typeMAX"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape
"
strides
"
data_formatNHWC"
auto_padVALID
�
#detector/yolo-v3-tiny/Conv_5/Conv2DConv%detector/yolo-v3-tiny/pool2_4/MaxPool"0
_output_shapes
:����������"
pads

    "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
5detector/yolo-v3-tiny/Conv_5/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_5/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%��'7"
dtype0

�
&detector/yolo-v3-tiny/Conv_5/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_5/BatchNorm/FusedBatchNorm"0
_output_shapes
:����������"
shape
����������"
dtype0

�
%detector/yolo-v3-tiny/pool2_5/MaxPoolPool&detector/yolo-v3-tiny/Conv_5/LeakyRelu"0
_output_shapes
:����������"
pads

      "
kernel_shape
"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
pooling_typeMAX"
dtype0

�
#detector/yolo-v3-tiny/Conv_6/Conv2DConv%detector/yolo-v3-tiny/pool2_5/MaxPool"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape

��"
strides
"
data_formatNHWC
�
5detector/yolo-v3-tiny/Conv_6/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_6/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%��'7"
dtype0
"0
_output_shapes
:����������
�
&detector/yolo-v3-tiny/Conv_6/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_6/BatchNorm/FusedBatchNorm"0
_output_shapes
:����������"
shape
����������"
dtype0

�
#detector/yolo-v3-tiny/Conv_7/Conv2DConv&detector/yolo-v3-tiny/Conv_6/LeakyRelu"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER
�
5detector/yolo-v3-tiny/Conv_7/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_7/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%��'7"
dtype0

�
&detector/yolo-v3-tiny/Conv_7/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_7/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
#detector/yolo-v3-tiny/Conv_8/Conv2DConv&detector/yolo-v3-tiny/Conv_7/LeakyRelu"0
_output_shapes
:����������"
pads

    "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
$detector/yolo-v3-tiny/Conv_10/Conv2DConv&detector/yolo-v3-tiny/Conv_7/LeakyRelu"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        
�
5detector/yolo-v3-tiny/Conv_8/BatchNorm/FusedBatchNorm	BatchNorm#detector/yolo-v3-tiny/Conv_8/Conv2D"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%��'7
�
6detector/yolo-v3-tiny/Conv_10/BatchNorm/FusedBatchNorm	BatchNorm$detector/yolo-v3-tiny/Conv_10/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%��'7"
dtype0
"0
_output_shapes
:����������
�
&detector/yolo-v3-tiny/Conv_8/LeakyRelu	LeakyRelu5detector/yolo-v3-tiny/Conv_8/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
'detector/yolo-v3-tiny/Conv_10/LeakyRelu	LeakyRelu6detector/yolo-v3-tiny/Conv_10/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
#detector/yolo-v3-tiny/Conv_9/Conv2DConv&detector/yolo-v3-tiny/Conv_8/LeakyRelu"0
_output_shapes
:����������"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
+detector/yolo-v3-tiny/ResizeNearestNeighborResizeNearestNeighbor'detector/yolo-v3-tiny/Conv_10/LeakyRelu0detector/yolo-v3-tiny/ResizeNearestNeighbor/size"
dtype0
"
size
"0
_output_shapes
:����������"
shape
����������
�
detector/yolo-v3-tiny/concat_3Concat+detector/yolo-v3-tiny/ResizeNearestNeighbor&detector/yolo-v3-tiny/Conv_4/LeakyRelu"

axis"
dtype0
"0
_output_shapes
:����������
�
$detector/yolo-v3-tiny/Conv_11/Conv2DConvdetector/yolo-v3-tiny/concat_3"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER
�
6detector/yolo-v3-tiny/Conv_11/BatchNorm/FusedBatchNorm	BatchNorm$detector/yolo-v3-tiny/Conv_11/Conv2D"
data_formatNHWC"
epsilon%��'7"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
'detector/yolo-v3-tiny/Conv_11/LeakyRelu	LeakyRelu6detector/yolo-v3-tiny/Conv_11/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
$detector/yolo-v3-tiny/Conv_12/Conv2DConv'detector/yolo-v3-tiny/Conv_11/LeakyRelu"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME