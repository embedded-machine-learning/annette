
s
Placeholder	DataInput"&
shape:�����������"1
_output_shapes
:�����������
i
x	DataInput"&
shape:�����������"1
_output_shapes
:�����������
�
resnet50v2/final_conv1_pad/PadPadPlaceholder"
mode
constant"
dtype0
"1
_output_shapes
:�����������"
pads

    
�
"resnet50v2/final_conv1_conv/Conv2DConvresnet50v2/final_conv1_pad/Pad"
kernel_shape
@"
strides
"
auto_padVALID"/
_output_shapes
:���������pp@"
shape
�����������"
paddingVALID"
pads

        "
data_formatNHWC"
use_bias("
dtype0

�
resnet50v2/final_pool1_pad/PadPad"resnet50v2/final_conv1_conv/Conv2D"
dtype0
"
pads

    "
mode
constant"/
_output_shapes
:���������rr@
�
#resnet50v2/final_pool1_pool/MaxPoolPoolresnet50v2/final_pool1_pad/Pad"
data_formatNHWC"
pooling_typeMAX"/
_output_shapes
:���������88@"
strides
"
pads

        "
auto_padVALID"
kernel_shape
"
dtype0

�
8resnet50v2/final_conv2_block1_preact_bn/FusedBatchNormV3	BatchNorm#resnet50v2/final_pool1_pool/MaxPool"
dtype0
"
scale("/
_output_shapes
:���������88@"

bias("
data_formatNHWC"
epsilon%��'7
�
.resnet50v2/final_conv2_block1_preact_relu/ReluRelu8resnet50v2/final_conv2_block1_preact_bn/FusedBatchNormV3"
shape
���������88@"/
_output_shapes
:���������88@"
dtype0

�
+resnet50v2/final_conv2_block1_0_conv/Conv2DConv.resnet50v2/final_conv2_block1_preact_relu/Relu"
kernel_shape	
@�"
dtype0
"
shape
���������88@"
data_formatNHWC"
strides
"
pads

        "
auto_padVALID"
use_bias("
paddingVALID"0
_output_shapes
:���������88�
�
+resnet50v2/final_conv2_block1_1_conv/Conv2DConv.resnet50v2/final_conv2_block1_preact_relu/Relu"
strides
"
dtype0
"
use_bias("
data_formatNHWC"
paddingVALID"
auto_padVALID"
kernel_shape
@@"
pads

        "
shape
���������88@"/
_output_shapes
:���������88@
�
)resnet50v2/final_conv2_block1_1_relu/ReluRelu+resnet50v2/final_conv2_block1_1_conv/Conv2D"
dtype0
"/
_output_shapes
:���������88@"
shape
���������88@
�
'resnet50v2/final_conv2_block1_2_pad/PadPad)resnet50v2/final_conv2_block1_1_relu/Relu"
dtype0
"/
_output_shapes
:���������::@"
mode
constant"
pads

    
�
+resnet50v2/final_conv2_block1_2_conv/Conv2DConv'resnet50v2/final_conv2_block1_2_pad/Pad"
shape
���������::@"/
_output_shapes
:���������88@"
auto_padVALID"
strides
"
use_bias("
kernel_shape
@@"
data_formatNHWC"
paddingVALID"
pads

        "
dtype0

�
)resnet50v2/final_conv2_block1_2_relu/ReluRelu+resnet50v2/final_conv2_block1_2_conv/Conv2D"/
_output_shapes
:���������88@"
shape
���������88@"
dtype0

�
+resnet50v2/final_conv2_block1_3_conv/Conv2DConv)resnet50v2/final_conv2_block1_2_relu/Relu"
use_bias("
data_formatNHWC"
dtype0
"
pads

        "
strides
"
auto_padVALID"
kernel_shape	
@�"0
_output_shapes
:���������88�"
shape
���������88@"
paddingVALID
�
%resnet50v2/final_conv2_block1_out/addAdd+resnet50v2/final_conv2_block1_0_conv/Conv2D+resnet50v2/final_conv2_block1_3_conv/Conv2D"0
_output_shapes
:���������88�"
dtype0

�
8resnet50v2/final_conv2_block2_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv2_block1_out/add"
data_formatNHWC"

bias("
scale("
dtype0
"
epsilon%��'7"0
_output_shapes
:���������88�
�
.resnet50v2/final_conv2_block2_preact_relu/ReluRelu8resnet50v2/final_conv2_block2_preact_bn/FusedBatchNormV3"0
_output_shapes
:���������88�"
shape
���������88�"
dtype0

�
+resnet50v2/final_conv2_block2_1_conv/Conv2DConv.resnet50v2/final_conv2_block2_preact_relu/Relu"
kernel_shape	
�@"/
_output_shapes
:���������88@"
use_bias("
shape
���������88�"
auto_padVALID"
pads

        "
data_formatNHWC"
paddingVALID"
strides
"
dtype0

�
)resnet50v2/final_conv2_block2_1_relu/ReluRelu+resnet50v2/final_conv2_block2_1_conv/Conv2D"/
_output_shapes
:���������88@"
shape
���������88@"
dtype0

�
'resnet50v2/final_conv2_block2_2_pad/PadPad)resnet50v2/final_conv2_block2_1_relu/Relu"/
_output_shapes
:���������::@"
dtype0
"
pads

    "
mode
constant
�
+resnet50v2/final_conv2_block2_2_conv/Conv2DConv'resnet50v2/final_conv2_block2_2_pad/Pad"
paddingVALID"
strides
"
auto_padVALID"/
_output_shapes
:���������88@"
shape
���������::@"
kernel_shape
@@"
dtype0
"
data_formatNHWC"
pads

        "
use_bias(
�
)resnet50v2/final_conv2_block2_2_relu/ReluRelu+resnet50v2/final_conv2_block2_2_conv/Conv2D"
dtype0
"
shape
���������88@"/
_output_shapes
:���������88@
�
+resnet50v2/final_conv2_block2_3_conv/Conv2DConv)resnet50v2/final_conv2_block2_2_relu/Relu"
pads

        "
auto_padVALID"0
_output_shapes
:���������88�"
shape
���������88@"
use_bias("
strides
"
paddingVALID"
data_formatNHWC"
kernel_shape	
@�"
dtype0

�
%resnet50v2/final_conv2_block2_out/addAdd%resnet50v2/final_conv2_block1_out/add+resnet50v2/final_conv2_block2_3_conv/Conv2D"
dtype0
"0
_output_shapes
:���������88�
�
&resnet50v2/final_max_pooling2d/MaxPoolPool%resnet50v2/final_conv2_block2_out/add"
auto_padVALID"
strides
"
pads

        "
pooling_typeMAX"
data_formatNHWC"
dtype0
"
kernel_shape
"0
_output_shapes
:����������
�
8resnet50v2/final_conv2_block3_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv2_block2_out/add"
dtype0
"
scale("

bias("
data_formatNHWC"
epsilon%��'7"0
_output_shapes
:���������88�
�
.resnet50v2/final_conv2_block3_preact_relu/ReluRelu8resnet50v2/final_conv2_block3_preact_bn/FusedBatchNormV3"
dtype0
"0
_output_shapes
:���������88�"
shape
���������88�
�
+resnet50v2/final_conv2_block3_1_conv/Conv2DConv.resnet50v2/final_conv2_block3_preact_relu/Relu"/
_output_shapes
:���������88@"
kernel_shape	
�@"
use_bias("
shape
���������88�"
strides
"
pads

        "
data_formatNHWC"
paddingVALID"
auto_padVALID"
dtype0

�
)resnet50v2/final_conv2_block3_1_relu/ReluRelu+resnet50v2/final_conv2_block3_1_conv/Conv2D"
shape
���������88@"/
_output_shapes
:���������88@"
dtype0

�
'resnet50v2/final_conv2_block3_2_pad/PadPad)resnet50v2/final_conv2_block3_1_relu/Relu"
mode
constant"/
_output_shapes
:���������::@"
dtype0
"
pads

    
�
+resnet50v2/final_conv2_block3_2_conv/Conv2DConv'resnet50v2/final_conv2_block3_2_pad/Pad"
data_formatNHWC"
kernel_shape
@@"
shape
���������::@"/
_output_shapes
:���������@"
paddingVALID"
auto_padVALID"
pads

        "
dtype0
"
strides
"
use_bias(
�
)resnet50v2/final_conv2_block3_2_relu/ReluRelu+resnet50v2/final_conv2_block3_2_conv/Conv2D"
shape
���������@"/
_output_shapes
:���������@"
dtype0

�
+resnet50v2/final_conv2_block3_3_conv/Conv2DConv)resnet50v2/final_conv2_block3_2_relu/Relu"
dtype0
"
data_formatNHWC"
pads

        "
strides
"
paddingVALID"
shape
���������@"
auto_padVALID"0
_output_shapes
:����������"
kernel_shape	
@�"
use_bias(
�
%resnet50v2/final_conv2_block3_out/addAdd&resnet50v2/final_max_pooling2d/MaxPool+resnet50v2/final_conv2_block3_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv3_block1_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv2_block3_out/add"
scale("
epsilon%��'7"

bias("
data_formatNHWC"0
_output_shapes
:����������"
dtype0

�
.resnet50v2/final_conv3_block1_preact_relu/ReluRelu8resnet50v2/final_conv3_block1_preact_bn/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
+resnet50v2/final_conv3_block1_0_conv/Conv2DConv.resnet50v2/final_conv3_block1_preact_relu/Relu"0
_output_shapes
:����������"
auto_padVALID"
use_bias("
strides
"
kernel_shape

��"
shape
����������"
data_formatNHWC"
pads

        "
paddingVALID"
dtype0

�
+resnet50v2/final_conv3_block1_1_conv/Conv2DConv.resnet50v2/final_conv3_block1_preact_relu/Relu"
strides
"
shape
����������"0
_output_shapes
:����������"
dtype0
"
data_formatNHWC"
paddingVALID"
use_bias("
auto_padVALID"
pads

        "
kernel_shape

��
�
)resnet50v2/final_conv3_block1_1_relu/ReluRelu+resnet50v2/final_conv3_block1_1_conv/Conv2D"0
_output_shapes
:����������"
shape
����������"
dtype0

�
'resnet50v2/final_conv3_block1_2_pad/PadPad)resnet50v2/final_conv3_block1_1_relu/Relu"0
_output_shapes
:����������"
pads

    "
mode
constant"
dtype0

�
+resnet50v2/final_conv3_block1_2_conv/Conv2DConv'resnet50v2/final_conv3_block1_2_pad/Pad"
auto_padVALID"
use_bias("
strides
"
dtype0
"
kernel_shape

��"
paddingVALID"
data_formatNHWC"
pads

        "0
_output_shapes
:����������"
shape
����������
�
)resnet50v2/final_conv3_block1_2_relu/ReluRelu+resnet50v2/final_conv3_block1_2_conv/Conv2D"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv3_block1_3_conv/Conv2DConv)resnet50v2/final_conv3_block1_2_relu/Relu"
shape
����������"0
_output_shapes
:����������"
strides
"
data_formatNHWC"
paddingVALID"
dtype0
"
auto_padVALID"
pads

        "
kernel_shape

��"
use_bias(
�
%resnet50v2/final_conv3_block1_out/addAdd+resnet50v2/final_conv3_block1_0_conv/Conv2D+resnet50v2/final_conv3_block1_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv3_block2_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv3_block1_out/add"
dtype0
"0
_output_shapes
:����������"
epsilon%��'7"
data_formatNHWC"

bias("
scale(
�
.resnet50v2/final_conv3_block2_preact_relu/ReluRelu8resnet50v2/final_conv3_block2_preact_bn/FusedBatchNormV3"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv3_block2_1_conv/Conv2DConv.resnet50v2/final_conv3_block2_preact_relu/Relu"
strides
"
paddingVALID"
kernel_shape

��"
dtype0
"0
_output_shapes
:����������"
use_bias("
data_formatNHWC"
shape
����������"
auto_padVALID"
pads

        
�
)resnet50v2/final_conv3_block2_1_relu/ReluRelu+resnet50v2/final_conv3_block2_1_conv/Conv2D"
shape
����������"
dtype0
"0
_output_shapes
:����������
�
'resnet50v2/final_conv3_block2_2_pad/PadPad)resnet50v2/final_conv3_block2_1_relu/Relu"
mode
constant"
pads

    "
dtype0
"0
_output_shapes
:����������
�
+resnet50v2/final_conv3_block2_2_conv/Conv2DConv'resnet50v2/final_conv3_block2_2_pad/Pad"
use_bias("0
_output_shapes
:����������"
auto_padVALID"
data_formatNHWC"
pads

        "
paddingVALID"
shape
����������"
strides
"
dtype0
"
kernel_shape

��
�
)resnet50v2/final_conv3_block2_2_relu/ReluRelu+resnet50v2/final_conv3_block2_2_conv/Conv2D"
shape
����������"0
_output_shapes
:����������"
dtype0

�
+resnet50v2/final_conv3_block2_3_conv/Conv2DConv)resnet50v2/final_conv3_block2_2_relu/Relu"
data_formatNHWC"
auto_padVALID"
shape
����������"
kernel_shape

��"
use_bias("
dtype0
"
paddingVALID"0
_output_shapes
:����������"
strides
"
pads

        
�
%resnet50v2/final_conv3_block2_out/addAdd%resnet50v2/final_conv3_block1_out/add+resnet50v2/final_conv3_block2_3_conv/Conv2D"0
_output_shapes
:����������"
dtype0

�
8resnet50v2/final_conv3_block3_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv3_block2_out/add"0
_output_shapes
:����������"
data_formatNHWC"

bias("
scale("
dtype0
"
epsilon%��'7
�
.resnet50v2/final_conv3_block3_preact_relu/ReluRelu8resnet50v2/final_conv3_block3_preact_bn/FusedBatchNormV3"
shape
����������"0
_output_shapes
:����������"
dtype0

�
+resnet50v2/final_conv3_block3_1_conv/Conv2DConv.resnet50v2/final_conv3_block3_preact_relu/Relu"
dtype0
"
data_formatNHWC"
kernel_shape

��"0
_output_shapes
:����������"
shape
����������"
pads

        "
paddingVALID"
auto_padVALID"
use_bias("
strides

�
)resnet50v2/final_conv3_block3_1_relu/ReluRelu+resnet50v2/final_conv3_block3_1_conv/Conv2D"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
'resnet50v2/final_conv3_block3_2_pad/PadPad)resnet50v2/final_conv3_block3_1_relu/Relu"
pads

    "0
_output_shapes
:����������"
dtype0
"
mode
constant
�
+resnet50v2/final_conv3_block3_2_conv/Conv2DConv'resnet50v2/final_conv3_block3_2_pad/Pad"
auto_padVALID"
kernel_shape

��"0
_output_shapes
:����������"
pads

        "
use_bias("
dtype0
"
data_formatNHWC"
paddingVALID"
shape
����������"
strides

�
)resnet50v2/final_conv3_block3_2_relu/ReluRelu+resnet50v2/final_conv3_block3_2_conv/Conv2D"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
+resnet50v2/final_conv3_block3_3_conv/Conv2DConv)resnet50v2/final_conv3_block3_2_relu/Relu"
paddingVALID"0
_output_shapes
:����������"
kernel_shape

��"
shape
����������"
strides
"
data_formatNHWC"
dtype0
"
pads

        "
use_bias("
auto_padVALID
�
%resnet50v2/final_conv3_block3_out/addAdd%resnet50v2/final_conv3_block2_out/add+resnet50v2/final_conv3_block3_3_conv/Conv2D"0
_output_shapes
:����������"
dtype0

�
(resnet50v2/final_max_pooling2d_1/MaxPoolPool%resnet50v2/final_conv3_block3_out/add"
strides
"
data_formatNHWC"
pads

        "
dtype0
"
auto_padVALID"0
_output_shapes
:����������"
pooling_typeMAX"
kernel_shape

�
8resnet50v2/final_conv3_block4_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv3_block3_out/add"

bias("
epsilon%��'7"
dtype0
"
data_formatNHWC"0
_output_shapes
:����������"
scale(
�
.resnet50v2/final_conv3_block4_preact_relu/ReluRelu8resnet50v2/final_conv3_block4_preact_bn/FusedBatchNormV3"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv3_block4_1_conv/Conv2DConv.resnet50v2/final_conv3_block4_preact_relu/Relu"
shape
����������"
data_formatNHWC"0
_output_shapes
:����������"
paddingVALID"
use_bias("
dtype0
"
kernel_shape

��"
pads

        "
auto_padVALID"
strides

�
)resnet50v2/final_conv3_block4_1_relu/ReluRelu+resnet50v2/final_conv3_block4_1_conv/Conv2D"0
_output_shapes
:����������"
dtype0
"
shape
����������
�
'resnet50v2/final_conv3_block4_2_pad/PadPad)resnet50v2/final_conv3_block4_1_relu/Relu"0
_output_shapes
:����������"
mode
constant"
dtype0
"
pads

    
�
+resnet50v2/final_conv3_block4_2_conv/Conv2DConv'resnet50v2/final_conv3_block4_2_pad/Pad"
auto_padVALID"
paddingVALID"
use_bias("
pads

        "0
_output_shapes
:����������"
data_formatNHWC"
shape
����������"
strides
"
dtype0
"
kernel_shape

��
�
)resnet50v2/final_conv3_block4_2_relu/ReluRelu+resnet50v2/final_conv3_block4_2_conv/Conv2D"
shape
����������"
dtype0
"0
_output_shapes
:����������
�
+resnet50v2/final_conv3_block4_3_conv/Conv2DConv)resnet50v2/final_conv3_block4_2_relu/Relu"
kernel_shape

��"
auto_padVALID"
shape
����������"
data_formatNHWC"
use_bias("
strides
"0
_output_shapes
:����������"
paddingVALID"
pads

        "
dtype0

�
%resnet50v2/final_conv3_block4_out/addAdd(resnet50v2/final_max_pooling2d_1/MaxPool+resnet50v2/final_conv3_block4_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv4_block1_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv3_block4_out/add"0
_output_shapes
:����������"
dtype0
"
epsilon%��'7"
scale("

bias("
data_formatNHWC
�
.resnet50v2/final_conv4_block1_preact_relu/ReluRelu8resnet50v2/final_conv4_block1_preact_bn/FusedBatchNormV3"
shape
����������"0
_output_shapes
:����������"
dtype0

�
+resnet50v2/final_conv4_block1_0_conv/Conv2DConv.resnet50v2/final_conv4_block1_preact_relu/Relu"
strides
"
dtype0
"
paddingVALID"
kernel_shape

��"
auto_padVALID"
use_bias("
shape
����������"0
_output_shapes
:����������"
pads

        "
data_formatNHWC
�
+resnet50v2/final_conv4_block1_1_conv/Conv2DConv.resnet50v2/final_conv4_block1_preact_relu/Relu"
paddingVALID"
strides
"
dtype0
"
data_formatNHWC"
use_bias("
shape
����������"
auto_padVALID"
kernel_shape

��"0
_output_shapes
:����������"
pads

        
�
)resnet50v2/final_conv4_block1_1_relu/ReluRelu+resnet50v2/final_conv4_block1_1_conv/Conv2D"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
'resnet50v2/final_conv4_block1_2_pad/PadPad)resnet50v2/final_conv4_block1_1_relu/Relu"
mode
constant"0
_output_shapes
:����������"
pads

    "
dtype0

�
+resnet50v2/final_conv4_block1_2_conv/Conv2DConv'resnet50v2/final_conv4_block1_2_pad/Pad"
strides
"
kernel_shape

��"
pads

        "
paddingVALID"
use_bias("
shape
����������"0
_output_shapes
:����������"
dtype0
"
data_formatNHWC"
auto_padVALID
�
)resnet50v2/final_conv4_block1_2_relu/ReluRelu+resnet50v2/final_conv4_block1_2_conv/Conv2D"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
+resnet50v2/final_conv4_block1_3_conv/Conv2DConv)resnet50v2/final_conv4_block1_2_relu/Relu"
paddingVALID"
strides
"
use_bias("
data_formatNHWC"
kernel_shape

��"
shape
����������"
dtype0
"0
_output_shapes
:����������"
pads

        "
auto_padVALID
�
%resnet50v2/final_conv4_block1_out/addAdd+resnet50v2/final_conv4_block1_0_conv/Conv2D+resnet50v2/final_conv4_block1_3_conv/Conv2D"0
_output_shapes
:����������"
dtype0

�
resnet50v2/shunt_conv2d/Conv2DConv%resnet50v2/final_conv4_block1_out/add"0
_output_shapes
:����������"
paddingSAME"
use_bias("
auto_pad
SAME_LOWER"
data_formatNHWC"
pads

        "
shape
����������"
kernel_shape

��"
strides
"
dtype0

�
resnet50v2/shunt_re_lu/Relu6Relu6resnet50v2/shunt_conv2d/Conv2D"0
_output_shapes
:����������"
dtype0

�
+resnet50v2/shunt_depthwise_conv2d/depthwiseDepthwiseConvresnet50v2/shunt_re_lu/Relu6"
use_bias("
kernel_shape	
�"0
_output_shapes
:����������"
data_formatNHWC"
pads

    "
auto_pad
SAME_LOWER"
dtype0
"
strides

�
resnet50v2/shunt_re_lu_1/Relu6Relu6+resnet50v2/shunt_depthwise_conv2d/depthwise"
dtype0
"0
_output_shapes
:����������
�
 resnet50v2/shunt_conv2d_1/Conv2DConvresnet50v2/shunt_re_lu_1/Relu6"
kernel_shape

��"
strides
"
pads

        "
paddingSAME"
dtype0
"
data_formatNHWC"
use_bias("0
_output_shapes
:����������"
shape
����������"
auto_pad
SAME_LOWER
�
(resnet50v2/final_max_pooling2d_2/MaxPoolPool resnet50v2/shunt_conv2d_1/Conv2D"
dtype0
"
pads

        "
data_formatNHWC"
kernel_shape
"
auto_padVALID"
strides
"0
_output_shapes
:����������"
pooling_typeMAX
�
8resnet50v2/final_conv4_block6_preact_bn/FusedBatchNormV3	BatchNorm resnet50v2/shunt_conv2d_1/Conv2D"
data_formatNHWC"
scale("

bias("
epsilon%��'7"
dtype0
"0
_output_shapes
:����������
�
.resnet50v2/final_conv4_block6_preact_relu/ReluRelu8resnet50v2/final_conv4_block6_preact_bn/FusedBatchNormV3"
shape
����������"
dtype0
"0
_output_shapes
:����������
�
+resnet50v2/final_conv4_block6_1_conv/Conv2DConv.resnet50v2/final_conv4_block6_preact_relu/Relu"
kernel_shape

��"
shape
����������"
pads

        "0
_output_shapes
:����������"
use_bias("
paddingVALID"
auto_padVALID"
strides
"
data_formatNHWC"
dtype0

�
)resnet50v2/final_conv4_block6_1_relu/ReluRelu+resnet50v2/final_conv4_block6_1_conv/Conv2D"0
_output_shapes
:����������"
dtype0
"
shape
����������
�
'resnet50v2/final_conv4_block6_2_pad/PadPad)resnet50v2/final_conv4_block6_1_relu/Relu"0
_output_shapes
:����������"
mode
constant"
dtype0
"
pads

    
�
+resnet50v2/final_conv4_block6_2_conv/Conv2DConv'resnet50v2/final_conv4_block6_2_pad/Pad"
paddingVALID"
kernel_shape

��"
auto_padVALID"
strides
"
shape
����������"0
_output_shapes
:����������"
data_formatNHWC"
dtype0
"
use_bias("
pads

        
�
)resnet50v2/final_conv4_block6_2_relu/ReluRelu+resnet50v2/final_conv4_block6_2_conv/Conv2D"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv4_block6_3_conv/Conv2DConv)resnet50v2/final_conv4_block6_2_relu/Relu"
strides
"
shape
����������"
paddingVALID"
kernel_shape

��"
dtype0
"
data_formatNHWC"
use_bias("
auto_padVALID"0
_output_shapes
:����������"
pads

        
�
%resnet50v2/final_conv4_block6_out/addAdd(resnet50v2/final_max_pooling2d_2/MaxPool+resnet50v2/final_conv4_block6_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv5_block1_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv4_block6_out/add"

bias("
dtype0
"0
_output_shapes
:����������"
data_formatNHWC"
epsilon%��'7"
scale(
�
.resnet50v2/final_conv5_block1_preact_relu/ReluRelu8resnet50v2/final_conv5_block1_preact_bn/FusedBatchNormV3"
shape
����������"0
_output_shapes
:����������"
dtype0

�
+resnet50v2/final_conv5_block1_0_conv/Conv2DConv.resnet50v2/final_conv5_block1_preact_relu/Relu"
paddingVALID"
data_formatNHWC"0
_output_shapes
:����������"
use_bias("
pads

        "
shape
����������"
dtype0
"
strides
"
kernel_shape

��"
auto_padVALID
�
+resnet50v2/final_conv5_block1_1_conv/Conv2DConv.resnet50v2/final_conv5_block1_preact_relu/Relu"
kernel_shape

��"
dtype0
"0
_output_shapes
:����������"
paddingVALID"
data_formatNHWC"
auto_padVALID"
pads

        "
shape
����������"
strides
"
use_bias(
�
)resnet50v2/final_conv5_block1_1_relu/ReluRelu+resnet50v2/final_conv5_block1_1_conv/Conv2D"0
_output_shapes
:����������"
shape
����������"
dtype0

�
'resnet50v2/final_conv5_block1_2_pad/PadPad)resnet50v2/final_conv5_block1_1_relu/Relu"
mode
constant"
pads

    "
dtype0
"0
_output_shapes
:���������		�
�
+resnet50v2/final_conv5_block1_2_conv/Conv2DConv'resnet50v2/final_conv5_block1_2_pad/Pad"
kernel_shape

��"
use_bias("
strides
"
paddingVALID"
auto_padVALID"
dtype0
"0
_output_shapes
:����������"
shape
���������		�"
data_formatNHWC"
pads

        
�
)resnet50v2/final_conv5_block1_2_relu/ReluRelu+resnet50v2/final_conv5_block1_2_conv/Conv2D"0
_output_shapes
:����������"
dtype0
"
shape
����������
�
+resnet50v2/final_conv5_block1_3_conv/Conv2DConv)resnet50v2/final_conv5_block1_2_relu/Relu"
use_bias("
kernel_shape

��"
auto_padVALID"
strides
"
paddingVALID"
pads

        "
dtype0
"
data_formatNHWC"0
_output_shapes
:����������"
shape
����������
�
%resnet50v2/final_conv5_block1_out/addAdd+resnet50v2/final_conv5_block1_0_conv/Conv2D+resnet50v2/final_conv5_block1_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv5_block2_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv5_block1_out/add"
scale("0
_output_shapes
:����������"
epsilon%��'7"
data_formatNHWC"

bias("
dtype0

�
.resnet50v2/final_conv5_block2_preact_relu/ReluRelu8resnet50v2/final_conv5_block2_preact_bn/FusedBatchNormV3"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv5_block2_1_conv/Conv2DConv.resnet50v2/final_conv5_block2_preact_relu/Relu"
use_bias("
strides
"
pads

        "
kernel_shape

��"
dtype0
"
auto_padVALID"0
_output_shapes
:����������"
shape
����������"
data_formatNHWC"
paddingVALID
�
)resnet50v2/final_conv5_block2_1_relu/ReluRelu+resnet50v2/final_conv5_block2_1_conv/Conv2D"0
_output_shapes
:����������"
shape
����������"
dtype0

�
'resnet50v2/final_conv5_block2_2_pad/PadPad)resnet50v2/final_conv5_block2_1_relu/Relu"0
_output_shapes
:���������		�"
mode
constant"
dtype0
"
pads

    
�
+resnet50v2/final_conv5_block2_2_conv/Conv2DConv'resnet50v2/final_conv5_block2_2_pad/Pad"
data_formatNHWC"
shape
���������		�"
auto_padVALID"
strides
"
pads

        "
dtype0
"0
_output_shapes
:����������"
use_bias("
kernel_shape

��"
paddingVALID
�
)resnet50v2/final_conv5_block2_2_relu/ReluRelu+resnet50v2/final_conv5_block2_2_conv/Conv2D"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
+resnet50v2/final_conv5_block2_3_conv/Conv2DConv)resnet50v2/final_conv5_block2_2_relu/Relu"0
_output_shapes
:����������"
kernel_shape

��"
paddingVALID"
data_formatNHWC"
shape
����������"
strides
"
use_bias("
auto_padVALID"
pads

        "
dtype0

�
%resnet50v2/final_conv5_block2_out/addAdd%resnet50v2/final_conv5_block1_out/add+resnet50v2/final_conv5_block2_3_conv/Conv2D"
dtype0
"0
_output_shapes
:����������
�
8resnet50v2/final_conv5_block3_preact_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv5_block2_out/add"0
_output_shapes
:����������"
epsilon%��'7"
scale("
dtype0
"
data_formatNHWC"

bias(
�
.resnet50v2/final_conv5_block3_preact_relu/ReluRelu8resnet50v2/final_conv5_block3_preact_bn/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������"
shape
����������
�
+resnet50v2/final_conv5_block3_1_conv/Conv2DConv.resnet50v2/final_conv5_block3_preact_relu/Relu"
use_bias("0
_output_shapes
:����������"
auto_padVALID"
data_formatNHWC"
pads

        "
kernel_shape

��"
strides
"
paddingVALID"
dtype0
"
shape
����������
�
)resnet50v2/final_conv5_block3_1_relu/ReluRelu+resnet50v2/final_conv5_block3_1_conv/Conv2D"0
_output_shapes
:����������"
shape
����������"
dtype0

�
'resnet50v2/final_conv5_block3_2_pad/PadPad)resnet50v2/final_conv5_block3_1_relu/Relu"
pads

    "
dtype0
"0
_output_shapes
:���������		�"
mode
constant
�
+resnet50v2/final_conv5_block3_2_conv/Conv2DConv'resnet50v2/final_conv5_block3_2_pad/Pad"0
_output_shapes
:����������"
shape
���������		�"
paddingVALID"
pads

        "
auto_padVALID"
data_formatNHWC"
use_bias("
kernel_shape

��"
dtype0
"
strides

�
)resnet50v2/final_conv5_block3_2_relu/ReluRelu+resnet50v2/final_conv5_block3_2_conv/Conv2D"
shape
����������"
dtype0
"0
_output_shapes
:����������
�
+resnet50v2/final_conv5_block3_3_conv/Conv2DConv)resnet50v2/final_conv5_block3_2_relu/Relu"
strides
"
dtype0
"
data_formatNHWC"
shape
����������"
use_bias("
auto_padVALID"
paddingVALID"0
_output_shapes
:����������"
kernel_shape

��"
pads

        
�
%resnet50v2/final_conv5_block3_out/addAdd%resnet50v2/final_conv5_block2_out/add+resnet50v2/final_conv5_block3_3_conv/Conv2D"0
_output_shapes
:����������"
dtype0

�
)resnet50v2/final_post_bn/FusedBatchNormV3	BatchNorm%resnet50v2/final_conv5_block3_out/add"
epsilon%��'7"
data_formatNHWC"0
_output_shapes
:����������"
dtype0
"
scale("

bias(
�
resnet50v2/final_post_relu/ReluRelu)resnet50v2/final_post_bn/FusedBatchNormV3"
dtype0
"
shape
����������"0
_output_shapes
:����������
�
resnet50v2/final_avg_pool/Mean
ReduceMeanresnet50v2/final_post_relu/Relu"
axes
"(
_output_shapes
:����������"
keepdims( "
dtype0

�
#resnet50v2/final_predictions/MatMulFullyConnectedresnet50v2/final_avg_pool/Mean"
dtype0
"
units�"<
_output_shapes*
(:����������:����������"
use_bias(
�
 resnet50v2/final_softmax/SoftmaxSoftmax#resnet50v2/final_predictions/MatMul"
shape
����������"(
_output_shapes
:����������"
dtype0
"	
dim