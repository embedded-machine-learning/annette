
s
Placeholder	DataInput"&
shape:�����������"1
_output_shapes
:�����������
�
8FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2DConvPlaceholder"
shape
�����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"1
_output_shapes
:����������� "
pads

      "
kernel_shape
 "
strides
"
data_formatNHWC
�
JFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm	BatchNorm8FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Conv2D"
dtype0
"1
_output_shapes
:����������� "

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
7FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6Relu6JFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������� 
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseDepthwiseConv7FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/Relu6"
kernel_shape
 "
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"1
_output_shapes
:����������� "
pads

    
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"1
_output_shapes
:����������� 
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������� 
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6"
shape
����������� "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"1
_output_shapes
:�����������@"
pads

        "
kernel_shape
 @"
strides
"
data_formatNHWC
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D"
epsilon%o�:"
dtype0
"1
_output_shapes
:�����������@"

bias("
scale("
data_formatNHWC
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:�����������@
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6"
kernel_shape
@"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������h@"
pads

      
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������h@
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������h@
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6"
kernel_shape	
@�"
strides
"
data_formatNHWC"
shape
����������h@"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"1
_output_shapes
:����������h�"
pads

        
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"1
_output_shapes
:����������h�"

bias("
scale(
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������h�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"1
_output_shapes
:����������h�"
pads

    
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise"
epsilon%o�:"
dtype0
"1
_output_shapes
:����������h�"

bias("
scale("
data_formatNHWC
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������h�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6"
dtype0
"1
_output_shapes
:����������h�"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������h�"
auto_pad
SAME_LOWER"
paddingSAME
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D"
dtype0
"1
_output_shapes
:����������h�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"1
_output_shapes
:����������h�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������P4�"
pads

      
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������P4�
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������P4�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6"
shape
���������P4�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������P4�"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������P4�"

bias("
scale("
data_formatNHWC
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������P4�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������P4�"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������P4�
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������P4�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������P4�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������P4�"
pads

        
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������P4�"

bias("
scale("
data_formatNHWC
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������P4�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6"
dtype0
"0
_output_shapes
:���������(�"
pads

      "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale(
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������(�"
pads

        
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������(�"
pads

    
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale(
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������(�"
pads

        
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������(�"
pads

    
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������(�"
pads

        
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
EFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������(�"
pads

    
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm	BatchNormEFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6"
dtype0
"0
_output_shapes
:���������(�"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME
�
TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm	BatchNormBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
AFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6Relu6TFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
FFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDepthwiseConvAFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������(�"
pads

    
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm	BatchNormFFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
CFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6"
dtype0
"0
_output_shapes
:���������(�"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm	BatchNormCFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale(
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
FFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDepthwiseConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:���������(�"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm	BatchNormFFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
CFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������(�"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������(�"
pads

        
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm	BatchNormCFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������(�"

bias("
scale(
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������(�
�
FFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDepthwiseConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������"
pads

      
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm	BatchNormFFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
CFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm	BatchNormCFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
FFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDepthwiseConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������"
pads

    
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm	BatchNormFFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
CFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��
�
UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm	BatchNormCFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
BFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6Relu6UFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
JFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2DConvBFeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER
�
\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/FusedBatchNorm	BatchNormJFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
IFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Relu6Relu6\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
MFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2DConvIFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_2_1x1_256/Relu6"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:���������
�"
pads

     "
kernel_shape

��
�
_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/FusedBatchNorm	BatchNormMFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������
�"

bias("
scale(
�
LFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6Relu6_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������
�
�
JFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2DConvLFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512/Relu6"
dtype0
"0
_output_shapes
:���������
�"
pads

        "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������
�"
auto_pad
SAME_LOWER"
paddingSAME
�
\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/FusedBatchNorm	BatchNormJFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:���������
�
�
IFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Relu6Relu6\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:���������
�
�
MFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2DConvIFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_3_1x1_128/Relu6"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

     "
kernel_shape

��"
strides
"
data_formatNHWC"
shape
���������
�"
auto_pad
SAME_LOWER
�
_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/FusedBatchNorm	BatchNormMFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
LFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6Relu6_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
JFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Conv2DConvLFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256/Relu6"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��
�
\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/FusedBatchNorm	BatchNormJFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
IFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Relu6Relu6\FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������
�
MFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Conv2DConvIFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_1_Conv2d_4_1x1_128/Relu6"
kernel_shape

��"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

     
�
_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/FusedBatchNorm	BatchNormMFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC
�
LFeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/Relu6Relu6_FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_4_3x3_s2_256/BatchNorm/FusedBatchNorm"
dtype0
"0
_output_shapes
:����������