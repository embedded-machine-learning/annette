
s
Placeholder	DataInput"1
_output_shapes
:�����������"&
shape:�����������
�
Conv1_pad/PadPadPlaceholder"
dtype0
"1
_output_shapes
:�����������"
pads

      "
mode
constant
�
Conv1/Conv2DConvConv1_pad/Pad"
paddingVALID"
dtype0
"/
_output_shapes
:���������pp"
pads

        "
kernel_shape
"
strides
"
data_formatNHWC"
shape
�����������"
auto_padVALID
�
bn_Conv1/FusedBatchNormV3	BatchNormConv1/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������pp
r
Conv1_relu/Relu6Relu6bn_Conv1/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������pp
�
!expanded_conv_depthwise/depthwiseDepthwiseConvConv1_relu/Relu6"
auto_pad
SAME_LOWER"
dtype0
"/
_output_shapes
:���������pp"
pads

    "
kernel_shape
"
strides
"
data_formatNHWC
�
+expanded_conv_depthwise_BN/FusedBatchNormV3	BatchNorm!expanded_conv_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������pp
�
"expanded_conv_depthwise_relu/Relu6Relu6+expanded_conv_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������pp
�
expanded_conv_project/Conv2DConv"expanded_conv_depthwise_relu/Relu6"
kernel_shape
"
strides
"
data_formatNHWC"
shape
���������pp"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������pp"
pads

        
�
)expanded_conv_project_BN/FusedBatchNormV3	BatchNormexpanded_conv_project/Conv2D"
dtype0
"/
_output_shapes
:���������pp"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_1_expand/Conv2DConv)expanded_conv_project_BN/FusedBatchNormV3"
kernel_shape
0"
strides
"
data_formatNHWC"
shape
���������pp"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������pp0"
pads

        
�
"block_1_expand_BN/FusedBatchNormV3	BatchNormblock_1_expand/Conv2D"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������pp0"

bias(
�
block_1_expand_relu/Relu6Relu6"block_1_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������pp0
�
block_1_pad/PadPadblock_1_expand_relu/Relu6"
dtype0
"/
_output_shapes
:���������qq0"
pads

      "
mode
constant
�
block_1_depthwise/depthwiseDepthwiseConvblock_1_pad/Pad"
kernel_shape
0"
strides
"
data_formatNHWC"
auto_padVALID"
dtype0
"/
_output_shapes
:���������880"
pads

        
�
%block_1_depthwise_BN/FusedBatchNormV3	BatchNormblock_1_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������880
�
block_1_depthwise_relu/Relu6Relu6%block_1_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������880
�
block_1_project/Conv2DConvblock_1_depthwise_relu/Relu6"
kernel_shape
0"
strides
"
data_formatNHWC"
shape
���������880"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������88"
pads

        
�
#block_1_project_BN/FusedBatchNormV3	BatchNormblock_1_project/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������88
�
block_2_expand/Conv2DConv#block_1_project_BN/FusedBatchNormV3"
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������88"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������88`"
pads

        
�
"block_2_expand_BN/FusedBatchNormV3	BatchNormblock_2_expand/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������88`"

bias("
scale(
�
block_2_expand_relu/Relu6Relu6"block_2_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������88`
�
block_2_depthwise/depthwiseDepthwiseConvblock_2_expand_relu/Relu6"
kernel_shape
`"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"/
_output_shapes
:���������88`"
pads

    
�
%block_2_depthwise_BN/FusedBatchNormV3	BatchNormblock_2_depthwise/depthwise"
dtype0
"/
_output_shapes
:���������88`"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_2_depthwise_relu/Relu6Relu6%block_2_depthwise_BN/FusedBatchNormV3"/
_output_shapes
:���������88`"
dtype0

�
block_2_project/Conv2DConvblock_2_depthwise_relu/Relu6"
shape
���������88`"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������88"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC
�
#block_2_project_BN/FusedBatchNormV3	BatchNormblock_2_project/Conv2D"
dtype0
"/
_output_shapes
:���������88"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_2_add/addAdd#block_1_project_BN/FusedBatchNormV3#block_2_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������88
�
block_3_expand/Conv2DConvblock_2_add/add"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������88"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������88`
�
"block_3_expand_BN/FusedBatchNormV3	BatchNormblock_3_expand/Conv2D"/
_output_shapes
:���������88`"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0

�
block_3_expand_relu/Relu6Relu6"block_3_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������88`
�
block_3_pad/PadPadblock_3_expand_relu/Relu6"
dtype0
"/
_output_shapes
:���������99`"
pads

      "
mode
constant
�
block_3_depthwise/depthwiseDepthwiseConvblock_3_pad/Pad"
dtype0
"/
_output_shapes
:���������`"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
auto_padVALID
�
%block_3_depthwise_BN/FusedBatchNormV3	BatchNormblock_3_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`
�
block_3_depthwise_relu/Relu6Relu6%block_3_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_3_project/Conv2DConvblock_3_depthwise_relu/Relu6"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������`"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������
�
#block_3_project_BN/FusedBatchNormV3	BatchNormblock_3_project/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������
�
block_4_expand/Conv2DConv#block_3_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������"
auto_pad
SAME_LOWER"
paddingSAME
�
"block_4_expand_BN/FusedBatchNormV3	BatchNormblock_4_expand/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`
�
block_4_expand_relu/Relu6Relu6"block_4_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_4_depthwise/depthwiseDepthwiseConvblock_4_expand_relu/Relu6"
pads

    "
kernel_shape
`"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"/
_output_shapes
:���������`
�
%block_4_depthwise_BN/FusedBatchNormV3	BatchNormblock_4_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`
�
block_4_depthwise_relu/Relu6Relu6%block_4_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_4_project/Conv2DConvblock_4_depthwise_relu/Relu6"
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������`"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������"
pads

        
�
#block_4_project_BN/FusedBatchNormV3	BatchNormblock_4_project/Conv2D"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������"

bias("
scale("
data_formatNHWC
�
block_4_add/addAdd#block_3_project_BN/FusedBatchNormV3#block_4_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������
�
block_5_expand/Conv2DConvblock_4_add/add"/
_output_shapes
:���������`"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
"block_5_expand_BN/FusedBatchNormV3	BatchNormblock_5_expand/Conv2D"
dtype0
"/
_output_shapes
:���������`"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_5_expand_relu/Relu6Relu6"block_5_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_5_depthwise/depthwiseDepthwiseConvblock_5_expand_relu/Relu6"
dtype0
"/
_output_shapes
:���������`"
pads

    "
kernel_shape
`"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER
�
%block_5_depthwise_BN/FusedBatchNormV3	BatchNormblock_5_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`
�
block_5_depthwise_relu/Relu6Relu6%block_5_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_5_project/Conv2DConvblock_5_depthwise_relu/Relu6"
shape
���������`"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC
�
#block_5_project_BN/FusedBatchNormV3	BatchNormblock_5_project/Conv2D"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������"

bias(
�
block_5_add/addAddblock_4_add/add#block_5_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������
�
block_6_expand/Conv2DConvblock_5_add/add"
kernel_shape
`"
strides
"
data_formatNHWC"
shape
���������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������`"
pads

        
�
"block_6_expand_BN/FusedBatchNormV3	BatchNormblock_6_expand/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`"

bias("
scale(
�
block_6_expand_relu/Relu6Relu6"block_6_expand_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_6_pad/PadPadblock_6_expand_relu/Relu6"
dtype0
"/
_output_shapes
:���������`"
pads

      "
mode
constant
�
block_6_depthwise/depthwiseDepthwiseConvblock_6_pad/Pad"
dtype0
"/
_output_shapes
:���������`"
pads

        "
kernel_shape
`"
strides
"
data_formatNHWC"
auto_padVALID
�
%block_6_depthwise_BN/FusedBatchNormV3	BatchNormblock_6_depthwise/depthwise"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������`"

bias(
�
block_6_depthwise_relu/Relu6Relu6%block_6_depthwise_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������`
�
block_6_project/Conv2DConvblock_6_depthwise_relu/Relu6"
kernel_shape
` "
strides
"
data_formatNHWC"
shape
���������`"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:��������� "
pads

        
�
#block_6_project_BN/FusedBatchNormV3	BatchNormblock_6_project/Conv2D"
epsilon%o�:"
dtype0
"/
_output_shapes
:��������� "

bias("
scale("
data_formatNHWC
�
block_7_expand/Conv2DConv#block_6_project_BN/FusedBatchNormV3"
kernel_shape	
 �"
strides
"
data_formatNHWC"
shape
��������� "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        
�
"block_7_expand_BN/FusedBatchNormV3	BatchNormblock_7_expand/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC
�
block_7_expand_relu/Relu6Relu6"block_7_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_7_depthwise/depthwiseDepthwiseConvblock_7_expand_relu/Relu6"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC
�
%block_7_depthwise_BN/FusedBatchNormV3	BatchNormblock_7_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_7_depthwise_relu/Relu6Relu6%block_7_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_7_project/Conv2DConvblock_7_depthwise_relu/Relu6"
paddingSAME"
dtype0
"/
_output_shapes
:��������� "
pads

        "
kernel_shape	
� "
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER
�
#block_7_project_BN/FusedBatchNormV3	BatchNormblock_7_project/Conv2D"
epsilon%o�:"
dtype0
"/
_output_shapes
:��������� "

bias("
scale("
data_formatNHWC
�
block_7_add/addAdd#block_6_project_BN/FusedBatchNormV3#block_7_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:��������� 
�
block_8_expand/Conv2DConvblock_7_add/add"
shape
��������� "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
 �"
strides
"
data_formatNHWC
�
"block_8_expand_BN/FusedBatchNormV3	BatchNormblock_8_expand/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC
�
block_8_expand_relu/Relu6Relu6"block_8_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_8_depthwise/depthwiseDepthwiseConvblock_8_expand_relu/Relu6"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER
�
%block_8_depthwise_BN/FusedBatchNormV3	BatchNormblock_8_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_8_depthwise_relu/Relu6Relu6%block_8_depthwise_BN/FusedBatchNormV3"0
_output_shapes
:����������"
dtype0

�
block_8_project/Conv2DConvblock_8_depthwise_relu/Relu6"
dtype0
"/
_output_shapes
:��������� "
pads

        "
kernel_shape	
� "
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME
�
#block_8_project_BN/FusedBatchNormV3	BatchNormblock_8_project/Conv2D"
dtype0
"/
_output_shapes
:��������� "

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_8_add/addAddblock_7_add/add#block_8_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:��������� 
�
block_9_expand/Conv2DConvblock_8_add/add"
strides
"
data_formatNHWC"
shape
��������� "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
 �
�
"block_9_expand_BN/FusedBatchNormV3	BatchNormblock_9_expand/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0

�
block_9_expand_relu/Relu6Relu6"block_9_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_9_depthwise/depthwiseDepthwiseConvblock_9_expand_relu/Relu6"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0

�
%block_9_depthwise_BN/FusedBatchNormV3	BatchNormblock_9_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_9_depthwise_relu/Relu6Relu6%block_9_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_9_project/Conv2DConvblock_9_depthwise_relu/Relu6"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:��������� "
pads

        "
kernel_shape	
� 
�
#block_9_project_BN/FusedBatchNormV3	BatchNormblock_9_project/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:��������� 
�
block_9_add/addAddblock_8_add/add#block_9_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:��������� 
�
block_10_expand/Conv2DConvblock_9_add/add"
kernel_shape	
 �"
strides
"
data_formatNHWC"
shape
��������� "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        
�
#block_10_expand_BN/FusedBatchNormV3	BatchNormblock_10_expand/Conv2D"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_10_expand_relu/Relu6Relu6#block_10_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_10_depthwise/depthwiseDepthwiseConvblock_10_expand_relu/Relu6"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0

�
&block_10_depthwise_BN/FusedBatchNormV3	BatchNormblock_10_depthwise/depthwise"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias(
�
block_10_depthwise_relu/Relu6Relu6&block_10_depthwise_BN/FusedBatchNormV3"0
_output_shapes
:����������"
dtype0

�
block_10_project/Conv2DConvblock_10_depthwise_relu/Relu6"
dtype0
"/
_output_shapes
:���������0"
pads

        "
kernel_shape	
�0"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME
�
$block_10_project_BN/FusedBatchNormV3	BatchNormblock_10_project/Conv2D"
dtype0
"/
_output_shapes
:���������0"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_11_expand/Conv2DConv$block_10_project_BN/FusedBatchNormV3"
shape
���������0"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
0�"
strides
"
data_formatNHWC
�
#block_11_expand_BN/FusedBatchNormV3	BatchNormblock_11_expand/Conv2D"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_11_expand_relu/Relu6Relu6#block_11_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_11_depthwise/depthwiseDepthwiseConvblock_11_expand_relu/Relu6"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0

�
&block_11_depthwise_BN/FusedBatchNormV3	BatchNormblock_11_depthwise/depthwise"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_11_depthwise_relu/Relu6Relu6&block_11_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_11_project/Conv2DConvblock_11_depthwise_relu/Relu6"/
_output_shapes
:���������0"
pads

        "
kernel_shape	
�0"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
$block_11_project_BN/FusedBatchNormV3	BatchNormblock_11_project/Conv2D"
dtype0
"/
_output_shapes
:���������0"

bias("
scale("
data_formatNHWC"
epsilon%o�:
�
block_11_add/addAdd$block_10_project_BN/FusedBatchNormV3$block_11_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������0
�
block_12_expand/Conv2DConvblock_11_add/add"
kernel_shape	
0�"
strides
"
data_formatNHWC"
shape
���������0"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        
�
#block_12_expand_BN/FusedBatchNormV3	BatchNormblock_12_expand/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0

�
block_12_expand_relu/Relu6Relu6#block_12_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_12_depthwise/depthwiseDepthwiseConvblock_12_expand_relu/Relu6"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER
�
&block_12_depthwise_BN/FusedBatchNormV3	BatchNormblock_12_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_12_depthwise_relu/Relu6Relu6&block_12_depthwise_BN/FusedBatchNormV3"0
_output_shapes
:����������"
dtype0

�
block_12_project/Conv2DConvblock_12_depthwise_relu/Relu6"
paddingSAME"
dtype0
"/
_output_shapes
:���������0"
pads

        "
kernel_shape	
�0"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER
�
$block_12_project_BN/FusedBatchNormV3	BatchNormblock_12_project/Conv2D"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������0
�
block_12_add/addAddblock_11_add/add$block_12_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������0
�
block_13_expand/Conv2DConvblock_12_add/add"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
0�"
strides
"
data_formatNHWC"
shape
���������0"
auto_pad
SAME_LOWER"
paddingSAME
�
#block_13_expand_BN/FusedBatchNormV3	BatchNormblock_13_expand/Conv2D"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC
�
block_13_expand_relu/Relu6Relu6#block_13_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_13_pad/PadPadblock_13_expand_relu/Relu6"0
_output_shapes
:����������"
pads

      "
mode
constant"
dtype0

�
block_13_depthwise/depthwiseDepthwiseConvblock_13_pad/Pad"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_padVALID"
dtype0
"0
_output_shapes
:����������"
pads

        
�
&block_13_depthwise_BN/FusedBatchNormV3	BatchNormblock_13_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_13_depthwise_relu/Relu6Relu6&block_13_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_13_project/Conv2DConvblock_13_depthwise_relu/Relu6"
dtype0
"/
_output_shapes
:���������P"
pads

        "
kernel_shape	
�P"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME
�
$block_13_project_BN/FusedBatchNormV3	BatchNormblock_13_project/Conv2D"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������P"

bias("
scale("
data_formatNHWC
�
block_14_expand/Conv2DConv$block_13_project_BN/FusedBatchNormV3"0
_output_shapes
:����������"
pads

        "
kernel_shape	
P�"
strides
"
data_formatNHWC"
shape
���������P"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
#block_14_expand_BN/FusedBatchNormV3	BatchNormblock_14_expand/Conv2D"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0

�
block_14_expand_relu/Relu6Relu6#block_14_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_14_depthwise/depthwiseDepthwiseConvblock_14_expand_relu/Relu6"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER
�
&block_14_depthwise_BN/FusedBatchNormV3	BatchNormblock_14_depthwise/depthwise"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale("
data_formatNHWC
�
block_14_depthwise_relu/Relu6Relu6&block_14_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_14_project/Conv2DConvblock_14_depthwise_relu/Relu6"/
_output_shapes
:���������P"
pads

        "
kernel_shape	
�P"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0

�
$block_14_project_BN/FusedBatchNormV3	BatchNormblock_14_project/Conv2D"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������P"

bias("
scale("
data_formatNHWC
�
block_14_add/addAdd$block_13_project_BN/FusedBatchNormV3$block_14_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������P
�
block_15_expand/Conv2DConvblock_14_add/add"
strides
"
data_formatNHWC"
shape
���������P"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
P�
�
#block_15_expand_BN/FusedBatchNormV3	BatchNormblock_15_expand/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
block_15_expand_relu/Relu6Relu6#block_15_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_15_depthwise/depthwiseDepthwiseConvblock_15_expand_relu/Relu6"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������"
pads

    "
kernel_shape	
�
�
&block_15_depthwise_BN/FusedBatchNormV3	BatchNormblock_15_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_15_depthwise_relu/Relu6Relu6&block_15_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_15_project/Conv2DConvblock_15_depthwise_relu/Relu6"
kernel_shape	
�P"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"/
_output_shapes
:���������P"
pads

        
�
$block_15_project_BN/FusedBatchNormV3	BatchNormblock_15_project/Conv2D"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"/
_output_shapes
:���������P"

bias(
�
block_15_add/addAddblock_14_add/add$block_15_project_BN/FusedBatchNormV3"
dtype0
"/
_output_shapes
:���������P
�
block_16_expand/Conv2DConvblock_15_add/add"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape	
P�"
strides
"
data_formatNHWC"
shape
���������P"
auto_pad
SAME_LOWER"
paddingSAME
�
#block_16_expand_BN/FusedBatchNormV3	BatchNormblock_16_expand/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
block_16_expand_relu/Relu6Relu6#block_16_expand_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_16_depthwise/depthwiseDepthwiseConvblock_16_expand_relu/Relu6"
kernel_shape	
�"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:����������"
pads

    
�
&block_16_depthwise_BN/FusedBatchNormV3	BatchNormblock_16_depthwise/depthwise"

bias("
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
�
block_16_depthwise_relu/Relu6Relu6&block_16_depthwise_BN/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������
�
block_16_project/Conv2DConvblock_16_depthwise_relu/Relu6"
strides
"
data_formatNHWC"
shape
����������"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"0
_output_shapes
:����������"
pads

        "
kernel_shape

��
�
$block_16_project_BN/FusedBatchNormV3	BatchNormblock_16_project/Conv2D"
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������"

bias("
scale(
�
Conv_1/Conv2DConv$block_16_project_BN/FusedBatchNormV3"
kernel_shape

��
"
strides
"
data_formatNHWC"
shape
����������"
auto_padVALID"
paddingVALID"
dtype0
"0
_output_shapes
:����������
"
pads

        
�
Conv_1_bn/FusedBatchNormV3	BatchNormConv_1/Conv2D"
scale("
data_formatNHWC"
epsilon%o�:"
dtype0
"0
_output_shapes
:����������
"

bias(
r
out_relu/Relu6Relu6Conv_1_bn/FusedBatchNormV3"
dtype0
"0
_output_shapes
:����������

�
global_average_pooling2d/Mean
ReduceMeanout_relu/Relu6"(
_output_shapes
:����������
"
axes
"
keepdims( "
dtype0

�
Logits/MatMulFullyConnectedglobal_average_pooling2d/Mean"
units�"
dtype0
"
use_bias("<
_output_shapes*
(:����������:����������
�
Logits/SoftmaxSoftmaxLogits/MatMul"	
dim"
dtype0
"(
_output_shapes
:����������"
shape
����������