
s
Placeholder	DataInput"1
_output_shapes
:˙˙˙˙˙˙˙˙˙"&
shape:˙˙˙˙˙˙˙˙˙
n
inputs	DataInput"1
_output_shapes
:˙˙˙˙˙˙˙˙˙"&
shape:˙˙˙˙˙˙˙˙˙
|
detector/truedivRealDivPlaceholderdetector/truediv/y"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
"detector/cspdarknet-53/Conv/Conv2DConvdetector/truediv"
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape
 "
paddingSAME"
strides
"
pads

    "
auto_pad
SAME_LOWER"
data_formatNHWC"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
é
6detector/cspdarknet-53/Conv/BatchNorm/FusedBatchNormV3	BatchNorm"detector/cspdarknet-53/Conv/Conv2D"
scale("1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
data_formatNHWC"

bias("
dtype0
"
epsilon%đ'7
Ń
$detector/cspdarknet-53/Conv/SoftplusSoftplus6detector/cspdarknet-53/Conv/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙ "1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "	
dim"
dtype0

Ź
 detector/cspdarknet-53/Conv/TanhTanh$detector/cspdarknet-53/Conv/Softplus"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
shape
˙˙˙˙˙˙˙˙˙ "
dtype0

Ŕ
detector/cspdarknet-53/Conv/mulMul6detector/cspdarknet-53/Conv/BatchNorm/FusedBatchNormV3 detector/cspdarknet-53/Conv/Tanh"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ź
detector/cspdarknet-53/PadPaddetector/cspdarknet-53/Conv/mul"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
mode
constant"
dtype0
"
pads

    
Ť
$detector/cspdarknet-53/Conv_1/Conv2DConvdetector/cspdarknet-53/Pad"
auto_padVALID"
pads

        "
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙ "
kernel_shape
 @"
data_formatNHWC"
paddingVALID"
strides

í
8detector/cspdarknet-53/Conv_1/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_1/Conv2D"
epsilon%đ'7"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
scale("
data_formatNHWC"

bias(
Ő
&detector/cspdarknet-53/Conv_1/SoftplusSoftplus8detector/cspdarknet-53/Conv_1/BatchNorm/FusedBatchNormV3"	
dim"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@
°
"detector/cspdarknet-53/Conv_1/TanhTanh&detector/cspdarknet-53/Conv_1/Softplus"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@
Ć
!detector/cspdarknet-53/Conv_1/mulMul8detector/cspdarknet-53/Conv_1/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_1/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

ś
$detector/cspdarknet-53/Conv_2/Conv2DConv!detector/cspdarknet-53/Conv_1/mul"
shape
˙˙˙˙˙˙˙˙˙@"
paddingSAME"
data_formatNHWC"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
pads

        "
auto_pad
SAME_LOWER"
strides
"
dtype0
"
kernel_shape
@@
ś
$detector/cspdarknet-53/Conv_3/Conv2DConv!detector/cspdarknet-53/Conv_1/mul"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@"
kernel_shape
@@"
data_formatNHWC"
strides
"
pads

        "1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
paddingSAME"
auto_pad
SAME_LOWER
í
8detector/cspdarknet-53/Conv_2/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_2/Conv2D"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
scale("
dtype0
"
epsilon%đ'7"

bias("
data_formatNHWC
í
8detector/cspdarknet-53/Conv_3/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_3/Conv2D"

bias("
dtype0
"
data_formatNHWC"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
epsilon%đ'7"
scale(
Ő
&detector/cspdarknet-53/Conv_2/SoftplusSoftplus8detector/cspdarknet-53/Conv_2/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@"	
dim"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ő
&detector/cspdarknet-53/Conv_3/SoftplusSoftplus8detector/cspdarknet-53/Conv_3/BatchNorm/FusedBatchNormV3"	
dim"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

°
"detector/cspdarknet-53/Conv_2/TanhTanh&detector/cspdarknet-53/Conv_2/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
°
"detector/cspdarknet-53/Conv_3/TanhTanh&detector/cspdarknet-53/Conv_3/Softplus"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

Ć
!detector/cspdarknet-53/Conv_2/mulMul8detector/cspdarknet-53/Conv_2/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_2/Tanh"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ć
!detector/cspdarknet-53/Conv_3/mulMul8detector/cspdarknet-53/Conv_3/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_3/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

ś
$detector/cspdarknet-53/Conv_4/Conv2DConv!detector/cspdarknet-53/Conv_3/mul"
paddingSAME"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
shape
˙˙˙˙˙˙˙˙˙@"
strides
"
data_formatNHWC"
kernel_shape
@ "
dtype0
"
auto_pad
SAME_LOWER"
pads

        
í
8detector/cspdarknet-53/Conv_4/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_4/Conv2D"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
dtype0
"
data_formatNHWC"
epsilon%đ'7"

bias("
scale(
Ő
&detector/cspdarknet-53/Conv_4/SoftplusSoftplus8detector/cspdarknet-53/Conv_4/BatchNorm/FusedBatchNormV3"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
shape
˙˙˙˙˙˙˙˙˙ "
dtype0
"	
dim
°
"detector/cspdarknet-53/Conv_4/TanhTanh&detector/cspdarknet-53/Conv_4/Softplus"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
shape
˙˙˙˙˙˙˙˙˙ "
dtype0

Ć
!detector/cspdarknet-53/Conv_4/mulMul8detector/cspdarknet-53/Conv_4/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_4/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙ "
dtype0

ś
$detector/cspdarknet-53/Conv_5/Conv2DConv!detector/cspdarknet-53/Conv_4/mul"
dtype0
"
kernel_shape
 @"
pads

    "
paddingSAME"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
strides
"
shape
˙˙˙˙˙˙˙˙˙ "
auto_pad
SAME_LOWER"
data_formatNHWC
í
8detector/cspdarknet-53/Conv_5/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_5/Conv2D"
data_formatNHWC"
dtype0
"

bias("1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
epsilon%đ'7"
scale(
Ő
&detector/cspdarknet-53/Conv_5/SoftplusSoftplus8detector/cspdarknet-53/Conv_5/BatchNorm/FusedBatchNormV3"	
dim"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

°
"detector/cspdarknet-53/Conv_5/TanhTanh&detector/cspdarknet-53/Conv_5/Softplus"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

Ć
!detector/cspdarknet-53/Conv_5/mulMul8detector/cspdarknet-53/Conv_5/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_5/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

§
detector/cspdarknet-53/addAdd!detector/cspdarknet-53/Conv_3/mul!detector/cspdarknet-53/Conv_5/mul"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ż
$detector/cspdarknet-53/Conv_6/Conv2DConvdetector/cspdarknet-53/add"
strides
"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
auto_pad
SAME_LOWER"
pads

        "
dtype0
"
kernel_shape
@@"
data_formatNHWC
í
8detector/cspdarknet-53/Conv_6/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_6/Conv2D"

bias("
scale("
data_formatNHWC"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
epsilon%đ'7
Ő
&detector/cspdarknet-53/Conv_6/SoftplusSoftplus8detector/cspdarknet-53/Conv_6/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@"	
dim"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
°
"detector/cspdarknet-53/Conv_6/TanhTanh&detector/cspdarknet-53/Conv_6/Softplus"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@
Ć
!detector/cspdarknet-53/Conv_6/mulMul8detector/cspdarknet-53/Conv_6/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_6/Tanh"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ş
detector/cspdarknet-53/concatConcat!detector/cspdarknet-53/Conv_6/mul!detector/cspdarknet-53/Conv_2/mul"

axis"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
dtype0

´
$detector/cspdarknet-53/Conv_7/Conv2DConvdetector/cspdarknet-53/concat"
dtype0
"
pads

        "
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape	
@"
paddingSAME"
data_formatNHWC"
auto_pad
SAME_LOWER"
strides
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
í
8detector/cspdarknet-53/Conv_7/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_7/Conv2D"
dtype0
"
data_formatNHWC"
scale("1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
epsilon%đ'7"

bias(
Ő
&detector/cspdarknet-53/Conv_7/SoftplusSoftplus8detector/cspdarknet-53/Conv_7/BatchNorm/FusedBatchNormV3"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"	
dim"
shape
˙˙˙˙˙˙˙˙˙@
°
"detector/cspdarknet-53/Conv_7/TanhTanh&detector/cspdarknet-53/Conv_7/Softplus"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Ć
!detector/cspdarknet-53/Conv_7/mulMul8detector/cspdarknet-53/Conv_7/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_7/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

°
detector/cspdarknet-53/Pad_1Pad!detector/cspdarknet-53/Conv_7/mul"
mode
constant"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
pads

    
Ż
$detector/cspdarknet-53/Conv_8/Conv2DConvdetector/cspdarknet-53/Pad_1"
data_formatNHWC"
dtype0
"
paddingVALID"
strides
"
shape
˙˙˙˙˙˙˙˙˙@"
pads

        "
kernel_shape	
@"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
auto_padVALID
î
8detector/cspdarknet-53/Conv_8/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_8/Conv2D"
data_formatNHWC"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
scale("

bias("
dtype0
"
epsilon%đ'7
×
&detector/cspdarknet-53/Conv_8/SoftplusSoftplus8detector/cspdarknet-53/Conv_8/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"	
dim"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
˛
"detector/cspdarknet-53/Conv_8/TanhTanh&detector/cspdarknet-53/Conv_8/Softplus"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
Ç
!detector/cspdarknet-53/Conv_8/mulMul8detector/cspdarknet-53/Conv_8/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_8/Tanh"
dtype0
"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
¸
$detector/cspdarknet-53/Conv_9/Conv2DConv!detector/cspdarknet-53/Conv_8/mul"
strides
"
paddingSAME"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
data_formatNHWC"
pads

        "
auto_pad
SAME_LOWER"
dtype0
"
kernel_shape	
@"
shape
˙˙˙˙˙˙˙˙˙
š
%detector/cspdarknet-53/Conv_10/Conv2DConv!detector/cspdarknet-53/Conv_8/mul"
dtype0
"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙"
paddingSAME"
kernel_shape	
@"
strides
"
data_formatNHWC"
pads

        "1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
í
8detector/cspdarknet-53/Conv_9/BatchNorm/FusedBatchNormV3	BatchNorm$detector/cspdarknet-53/Conv_9/Conv2D"
scale("

bias("
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
data_formatNHWC"
epsilon%đ'7
ď
9detector/cspdarknet-53/Conv_10/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_10/Conv2D"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
scale("
dtype0
"
epsilon%đ'7"
data_formatNHWC"

bias(
Ő
&detector/cspdarknet-53/Conv_9/SoftplusSoftplus8detector/cspdarknet-53/Conv_9/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@
×
'detector/cspdarknet-53/Conv_10/SoftplusSoftplus9detector/cspdarknet-53/Conv_10/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"	
dim"
dtype0

°
"detector/cspdarknet-53/Conv_9/TanhTanh&detector/cspdarknet-53/Conv_9/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
˛
#detector/cspdarknet-53/Conv_10/TanhTanh'detector/cspdarknet-53/Conv_10/Softplus"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@
Ć
!detector/cspdarknet-53/Conv_9/mulMul8detector/cspdarknet-53/Conv_9/BatchNorm/FusedBatchNormV3"detector/cspdarknet-53/Conv_9/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

É
"detector/cspdarknet-53/Conv_10/mulMul9detector/cspdarknet-53/Conv_10/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_10/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

¸
%detector/cspdarknet-53/Conv_11/Conv2DConv"detector/cspdarknet-53/Conv_10/mul"
kernel_shape
@@"
paddingSAME"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
pads

        "
shape
˙˙˙˙˙˙˙˙˙@"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"
strides

ď
9detector/cspdarknet-53/Conv_11/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_11/Conv2D"
epsilon%đ'7"
data_formatNHWC"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
scale("

bias(
×
'detector/cspdarknet-53/Conv_11/SoftplusSoftplus9detector/cspdarknet-53/Conv_11/BatchNorm/FusedBatchNormV3"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"	
dim"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0

˛
#detector/cspdarknet-53/Conv_11/TanhTanh'detector/cspdarknet-53/Conv_11/Softplus"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
É
"detector/cspdarknet-53/Conv_11/mulMul9detector/cspdarknet-53/Conv_11/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_11/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

¸
%detector/cspdarknet-53/Conv_12/Conv2DConv"detector/cspdarknet-53/Conv_11/mul"
dtype0
"
strides
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@"
kernel_shape
@@"
data_formatNHWC"
pads

    "
paddingSAME"
auto_pad
SAME_LOWER
ď
9detector/cspdarknet-53/Conv_12/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_12/Conv2D"
epsilon%đ'7"
data_formatNHWC"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
scale("

bias(
×
'detector/cspdarknet-53/Conv_12/SoftplusSoftplus9detector/cspdarknet-53/Conv_12/BatchNorm/FusedBatchNormV3"	
dim"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@
˛
#detector/cspdarknet-53/Conv_12/TanhTanh'detector/cspdarknet-53/Conv_12/Softplus"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
É
"detector/cspdarknet-53/Conv_12/mulMul9detector/cspdarknet-53/Conv_12/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_12/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

Ť
detector/cspdarknet-53/add_1Add"detector/cspdarknet-53/Conv_10/mul"detector/cspdarknet-53/Conv_12/mul"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

˛
%detector/cspdarknet-53/Conv_13/Conv2DConvdetector/cspdarknet-53/add_1"
paddingSAME"
auto_pad
SAME_LOWER"
pads

        "
strides
"
kernel_shape
@@"
data_formatNHWC"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ď
9detector/cspdarknet-53/Conv_13/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_13/Conv2D"
data_formatNHWC"
scale("
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
epsilon%đ'7"

bias(
×
'detector/cspdarknet-53/Conv_13/SoftplusSoftplus9detector/cspdarknet-53/Conv_13/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@"	
dim"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
˛
#detector/cspdarknet-53/Conv_13/TanhTanh'detector/cspdarknet-53/Conv_13/Softplus"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@
É
"detector/cspdarknet-53/Conv_13/mulMul9detector/cspdarknet-53/Conv_13/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_13/Tanh"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
¸
%detector/cspdarknet-53/Conv_14/Conv2DConv"detector/cspdarknet-53/Conv_13/mul"
data_formatNHWC"
strides
"
kernel_shape
@@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
auto_pad
SAME_LOWER"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0
"
pads

    
ď
9detector/cspdarknet-53/Conv_14/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_14/Conv2D"
epsilon%đ'7"
dtype0
"
scale("1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
data_formatNHWC"

bias(
×
'detector/cspdarknet-53/Conv_14/SoftplusSoftplus9detector/cspdarknet-53/Conv_14/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"	
dim
˛
#detector/cspdarknet-53/Conv_14/TanhTanh'detector/cspdarknet-53/Conv_14/Softplus"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0

É
"detector/cspdarknet-53/Conv_14/mulMul9detector/cspdarknet-53/Conv_14/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_14/Tanh"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

Ľ
detector/cspdarknet-53/add_2Adddetector/cspdarknet-53/add_1"detector/cspdarknet-53/Conv_14/mul"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
˛
%detector/cspdarknet-53/Conv_15/Conv2DConvdetector/cspdarknet-53/add_2"
data_formatNHWC"
pads

        "
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
auto_pad
SAME_LOWER"
paddingSAME"
strides
"
dtype0
"
kernel_shape
@@
ď
9detector/cspdarknet-53/Conv_15/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_15/Conv2D"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"

bias("
dtype0
"
data_formatNHWC"
epsilon%đ'7"
scale(
×
'detector/cspdarknet-53/Conv_15/SoftplusSoftplus9detector/cspdarknet-53/Conv_15/BatchNorm/FusedBatchNormV3"	
dim"
shape
˙˙˙˙˙˙˙˙˙@"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
dtype0

˛
#detector/cspdarknet-53/Conv_15/TanhTanh'detector/cspdarknet-53/Conv_15/Softplus"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@"
shape
˙˙˙˙˙˙˙˙˙@"
dtype0

É
"detector/cspdarknet-53/Conv_15/mulMul9detector/cspdarknet-53/Conv_15/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_15/Tanh"
dtype0
"1
_output_shapes
:˙˙˙˙˙˙˙˙˙@
˝
detector/cspdarknet-53/concat_1Concat"detector/cspdarknet-53/Conv_15/mul!detector/cspdarknet-53/Conv_9/mul"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"

axis"
dtype0

š
%detector/cspdarknet-53/Conv_16/Conv2DConvdetector/cspdarknet-53/concat_1"
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"
pads

        "2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙"
strides
"
kernel_shape


đ
9detector/cspdarknet-53/Conv_16/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_16/Conv2D"
data_formatNHWC"
epsilon%đ'7"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
scale("
dtype0
"

bias(
Ů
'detector/cspdarknet-53/Conv_16/SoftplusSoftplus9detector/cspdarknet-53/Conv_16/BatchNorm/FusedBatchNormV3"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"	
dim
´
#detector/cspdarknet-53/Conv_16/TanhTanh'detector/cspdarknet-53/Conv_16/Softplus"
dtype0
"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
Ę
"detector/cspdarknet-53/Conv_16/mulMul9detector/cspdarknet-53/Conv_16/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_16/Tanh"
dtype0
"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
˛
detector/cspdarknet-53/Pad_2Pad"detector/cspdarknet-53/Conv_16/mul"2
_output_shapes 
:˙˙˙˙˙˙˙˙˙"
dtype0
"
pads

    "
mode
constant
°
%detector/cspdarknet-53/Conv_17/Conv2DConvdetector/cspdarknet-53/Pad_2"
dtype0
"
strides
"
shape
˙˙˙˙˙˙˙˙˙"
paddingVALID"
pads

        "
kernel_shape

"
data_formatNHWC"
auto_padVALID"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_17/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_17/Conv2D"
scale("

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"
data_formatNHWC"
dtype0

Ő
'detector/cspdarknet-53/Conv_17/SoftplusSoftplus9detector/cspdarknet-53/Conv_17/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_17/TanhTanh'detector/cspdarknet-53/Conv_17/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_17/mulMul9detector/cspdarknet-53/Conv_17/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_17/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_18/Conv2DConv"detector/cspdarknet-53/Conv_17/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙@@"
paddingSAME"
data_formatNHWC"
dtype0
"
kernel_shape

"
pads

        "
strides

¸
%detector/cspdarknet-53/Conv_19/Conv2DConv"detector/cspdarknet-53/Conv_17/mul"
auto_pad
SAME_LOWER"
data_formatNHWC"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"
pads

        "
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
strides
"
kernel_shape


î
9detector/cspdarknet-53/Conv_18/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_18/Conv2D"

bias("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale("
data_formatNHWC"
epsilon%đ'7
î
9detector/cspdarknet-53/Conv_19/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_19/Conv2D"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale("
epsilon%đ'7"
dtype0
"

bias(
Ő
'detector/cspdarknet-53/Conv_18/SoftplusSoftplus9detector/cspdarknet-53/Conv_18/BatchNorm/FusedBatchNormV3"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Ő
'detector/cspdarknet-53/Conv_19/SoftplusSoftplus9detector/cspdarknet-53/Conv_19/BatchNorm/FusedBatchNormV3"	
dim"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_18/TanhTanh'detector/cspdarknet-53/Conv_18/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_19/TanhTanh'detector/cspdarknet-53/Conv_19/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_18/mulMul9detector/cspdarknet-53/Conv_18/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_18/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_19/mulMul9detector/cspdarknet-53/Conv_19/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_19/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_20/Conv2DConv"detector/cspdarknet-53/Conv_19/mul"
pads

        "
dtype0
"
auto_pad
SAME_LOWER"
paddingSAME"
strides
"
kernel_shape

"
data_formatNHWC"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_20/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_20/Conv2D"
scale("
dtype0
"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_20/SoftplusSoftplus9detector/cspdarknet-53/Conv_20/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_20/TanhTanh'detector/cspdarknet-53/Conv_20/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_20/mulMul9detector/cspdarknet-53/Conv_20/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_20/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¸
%detector/cspdarknet-53/Conv_21/Conv2DConv"detector/cspdarknet-53/Conv_20/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
pads

    "
auto_pad
SAME_LOWER"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
kernel_shape

"
strides

î
9detector/cspdarknet-53/Conv_21/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_21/Conv2D"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"

bias("
scale("
data_formatNHWC"
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_21/SoftplusSoftplus9detector/cspdarknet-53/Conv_21/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_21/TanhTanh'detector/cspdarknet-53/Conv_21/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_21/mulMul9detector/cspdarknet-53/Conv_21/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_21/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

Ş
detector/cspdarknet-53/add_3Add"detector/cspdarknet-53/Conv_19/mul"detector/cspdarknet-53/Conv_21/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

˛
%detector/cspdarknet-53/Conv_22/Conv2DConvdetector/cspdarknet-53/add_3"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
paddingSAME"
pads

        "
dtype0
"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_22/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_22/Conv2D"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale("
dtype0
"
data_formatNHWC"
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_22/SoftplusSoftplus9detector/cspdarknet-53/Conv_22/BatchNorm/FusedBatchNormV3"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_22/TanhTanh'detector/cspdarknet-53/Conv_22/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_22/mulMul9detector/cspdarknet-53/Conv_22/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_22/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¸
%detector/cspdarknet-53/Conv_23/Conv2DConv"detector/cspdarknet-53/Conv_22/mul"
pads

    "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"
paddingSAME"
strides
"
kernel_shape

"
data_formatNHWC"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_23/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_23/Conv2D"
dtype0
"
data_formatNHWC"

bias("
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ő
'detector/cspdarknet-53/Conv_23/SoftplusSoftplus9detector/cspdarknet-53/Conv_23/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_23/TanhTanh'detector/cspdarknet-53/Conv_23/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_23/mulMul9detector/cspdarknet-53/Conv_23/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_23/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¤
detector/cspdarknet-53/add_4Adddetector/cspdarknet-53/add_3"detector/cspdarknet-53/Conv_23/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
˛
%detector/cspdarknet-53/Conv_24/Conv2DConvdetector/cspdarknet-53/add_4"
strides
"
dtype0
"
pads

        "
shape
˙˙˙˙˙˙˙˙˙@@"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
auto_pad
SAME_LOWER"
data_formatNHWC
î
9detector/cspdarknet-53/Conv_24/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_24/Conv2D"
scale("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"

bias("
epsilon%đ'7"
data_formatNHWC
Ő
'detector/cspdarknet-53/Conv_24/SoftplusSoftplus9detector/cspdarknet-53/Conv_24/BatchNorm/FusedBatchNormV3"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

°
#detector/cspdarknet-53/Conv_24/TanhTanh'detector/cspdarknet-53/Conv_24/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_24/mulMul9detector/cspdarknet-53/Conv_24/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_24/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_25/Conv2DConv"detector/cspdarknet-53/Conv_24/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
strides
"
paddingSAME"
kernel_shape

"
auto_pad
SAME_LOWER"
dtype0
"
pads

    "
data_formatNHWC
î
9detector/cspdarknet-53/Conv_25/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_25/Conv2D"
epsilon%đ'7"
data_formatNHWC"

bias("
scale("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ő
'detector/cspdarknet-53/Conv_25/SoftplusSoftplus9detector/cspdarknet-53/Conv_25/BatchNorm/FusedBatchNormV3"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_25/TanhTanh'detector/cspdarknet-53/Conv_25/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_25/mulMul9detector/cspdarknet-53/Conv_25/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_25/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¤
detector/cspdarknet-53/add_5Adddetector/cspdarknet-53/add_4"detector/cspdarknet-53/Conv_25/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
˛
%detector/cspdarknet-53/Conv_26/Conv2DConvdetector/cspdarknet-53/add_5"
kernel_shape

"
strides
"
auto_pad
SAME_LOWER"
dtype0
"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
pads

        "
shape
˙˙˙˙˙˙˙˙˙@@"
paddingSAME
î
9detector/cspdarknet-53/Conv_26/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_26/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
epsilon%đ'7"
data_formatNHWC"
scale("

bias(
Ő
'detector/cspdarknet-53/Conv_26/SoftplusSoftplus9detector/cspdarknet-53/Conv_26/BatchNorm/FusedBatchNormV3"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

°
#detector/cspdarknet-53/Conv_26/TanhTanh'detector/cspdarknet-53/Conv_26/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_26/mulMul9detector/cspdarknet-53/Conv_26/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_26/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_27/Conv2DConv"detector/cspdarknet-53/Conv_26/mul"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
pads

    "
strides
"
data_formatNHWC"
paddingSAME"
kernel_shape

"
dtype0

î
9detector/cspdarknet-53/Conv_27/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_27/Conv2D"

bias("
data_formatNHWC"
dtype0
"
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ő
'detector/cspdarknet-53/Conv_27/SoftplusSoftplus9detector/cspdarknet-53/Conv_27/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_27/TanhTanh'detector/cspdarknet-53/Conv_27/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_27/mulMul9detector/cspdarknet-53/Conv_27/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_27/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¤
detector/cspdarknet-53/add_6Adddetector/cspdarknet-53/add_5"detector/cspdarknet-53/Conv_27/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

˛
%detector/cspdarknet-53/Conv_28/Conv2DConvdetector/cspdarknet-53/add_6"
auto_pad
SAME_LOWER"
strides
"
data_formatNHWC"
kernel_shape

"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
paddingSAME"
dtype0

î
9detector/cspdarknet-53/Conv_28/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_28/Conv2D"
scale("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"

bias("
epsilon%đ'7"
dtype0

Ő
'detector/cspdarknet-53/Conv_28/SoftplusSoftplus9detector/cspdarknet-53/Conv_28/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_28/TanhTanh'detector/cspdarknet-53/Conv_28/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_28/mulMul9detector/cspdarknet-53/Conv_28/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_28/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¸
%detector/cspdarknet-53/Conv_29/Conv2DConv"detector/cspdarknet-53/Conv_28/mul"
auto_pad
SAME_LOWER"
kernel_shape

"
data_formatNHWC"
dtype0
"
strides
"
pads

    "
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_29/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_29/Conv2D"
dtype0
"
epsilon%đ'7"

bias("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale(
Ő
'detector/cspdarknet-53/Conv_29/SoftplusSoftplus9detector/cspdarknet-53/Conv_29/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

°
#detector/cspdarknet-53/Conv_29/TanhTanh'detector/cspdarknet-53/Conv_29/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_29/mulMul9detector/cspdarknet-53/Conv_29/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_29/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¤
detector/cspdarknet-53/add_7Adddetector/cspdarknet-53/add_6"detector/cspdarknet-53/Conv_29/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

˛
%detector/cspdarknet-53/Conv_30/Conv2DConvdetector/cspdarknet-53/add_7"
auto_pad
SAME_LOWER"
data_formatNHWC"
strides
"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
paddingSAME"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_30/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_30/Conv2D"
data_formatNHWC"

bias("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"
scale(
Ő
'detector/cspdarknet-53/Conv_30/SoftplusSoftplus9detector/cspdarknet-53/Conv_30/BatchNorm/FusedBatchNormV3"	
dim"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_30/TanhTanh'detector/cspdarknet-53/Conv_30/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_30/mulMul9detector/cspdarknet-53/Conv_30/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_30/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_31/Conv2DConv"detector/cspdarknet-53/Conv_30/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
paddingSAME"
pads

    "
auto_pad
SAME_LOWER"
strides
"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"
kernel_shape

"
data_formatNHWC
î
9detector/cspdarknet-53/Conv_31/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_31/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"

bias("
scale("
epsilon%đ'7"
data_formatNHWC
Ő
'detector/cspdarknet-53/Conv_31/SoftplusSoftplus9detector/cspdarknet-53/Conv_31/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_31/TanhTanh'detector/cspdarknet-53/Conv_31/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_31/mulMul9detector/cspdarknet-53/Conv_31/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_31/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¤
detector/cspdarknet-53/add_8Adddetector/cspdarknet-53/add_7"detector/cspdarknet-53/Conv_31/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

˛
%detector/cspdarknet-53/Conv_32/Conv2DConvdetector/cspdarknet-53/add_8"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
paddingSAME"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
strides
"
pads

        
î
9detector/cspdarknet-53/Conv_32/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_32/Conv2D"
epsilon%đ'7"
dtype0
"

bias("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale(
Ő
'detector/cspdarknet-53/Conv_32/SoftplusSoftplus9detector/cspdarknet-53/Conv_32/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_32/TanhTanh'detector/cspdarknet-53/Conv_32/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_32/mulMul9detector/cspdarknet-53/Conv_32/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_32/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¸
%detector/cspdarknet-53/Conv_33/Conv2DConv"detector/cspdarknet-53/Conv_32/mul"
data_formatNHWC"
strides
"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙@@"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
dtype0
"
paddingSAME
î
9detector/cspdarknet-53/Conv_33/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_33/Conv2D"
epsilon%đ'7"

bias("
dtype0
"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale(
Ő
'detector/cspdarknet-53/Conv_33/SoftplusSoftplus9detector/cspdarknet-53/Conv_33/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_33/TanhTanh'detector/cspdarknet-53/Conv_33/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_33/mulMul9detector/cspdarknet-53/Conv_33/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_33/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
¤
detector/cspdarknet-53/add_9Adddetector/cspdarknet-53/add_8"detector/cspdarknet-53/Conv_33/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

˛
%detector/cspdarknet-53/Conv_34/Conv2DConvdetector/cspdarknet-53/add_9"
strides
"
dtype0
"
paddingSAME"
data_formatNHWC"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
pads

        "
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_34/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_34/Conv2D"
data_formatNHWC"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"
dtype0
"
scale(
Ő
'detector/cspdarknet-53/Conv_34/SoftplusSoftplus9detector/cspdarknet-53/Conv_34/BatchNorm/FusedBatchNormV3"	
dim"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_34/TanhTanh'detector/cspdarknet-53/Conv_34/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_34/mulMul9detector/cspdarknet-53/Conv_34/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_34/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

¸
%detector/cspdarknet-53/Conv_35/Conv2DConv"detector/cspdarknet-53/Conv_34/mul"
data_formatNHWC"
pads

    "
paddingSAME"
dtype0
"
auto_pad
SAME_LOWER"
kernel_shape

"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_35/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_35/Conv2D"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"

bias("
scale("
dtype0

Ő
'detector/cspdarknet-53/Conv_35/SoftplusSoftplus9detector/cspdarknet-53/Conv_35/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@
°
#detector/cspdarknet-53/Conv_35/TanhTanh'detector/cspdarknet-53/Conv_35/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

Č
"detector/cspdarknet-53/Conv_35/mulMul9detector/cspdarknet-53/Conv_35/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_35/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ľ
detector/cspdarknet-53/add_10Adddetector/cspdarknet-53/add_9"detector/cspdarknet-53/Conv_35/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

ł
%detector/cspdarknet-53/Conv_36/Conv2DConvdetector/cspdarknet-53/add_10"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙@@"
strides
"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
data_formatNHWC"
dtype0
"
pads

        
î
9detector/cspdarknet-53/Conv_36/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_36/Conv2D"
epsilon%đ'7"
data_formatNHWC"

bias("
dtype0
"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Ő
'detector/cspdarknet-53/Conv_36/SoftplusSoftplus9detector/cspdarknet-53/Conv_36/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_36/TanhTanh'detector/cspdarknet-53/Conv_36/Softplus"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_36/mulMul9detector/cspdarknet-53/Conv_36/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_36/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ź
detector/cspdarknet-53/concat_2Concat"detector/cspdarknet-53/Conv_36/mul"detector/cspdarknet-53/Conv_18/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"

axis
ľ
%detector/cspdarknet-53/Conv_37/Conv2DConvdetector/cspdarknet-53/concat_2"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
strides
"
auto_pad
SAME_LOWER"
dtype0
"
pads

        "
shape
˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
paddingSAME"
kernel_shape


î
9detector/cspdarknet-53/Conv_37/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_37/Conv2D"
data_formatNHWC"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"
dtype0
"
scale(
Ő
'detector/cspdarknet-53/Conv_37/SoftplusSoftplus9detector/cspdarknet-53/Conv_37/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"	
dim"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0

°
#detector/cspdarknet-53/Conv_37/TanhTanh'detector/cspdarknet-53/Conv_37/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
Č
"detector/cspdarknet-53/Conv_37/mulMul9detector/cspdarknet-53/Conv_37/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_37/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
°
detector/cspdarknet-53/Pad_3Pad"detector/cspdarknet-53/Conv_37/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙BB"
mode
constant"
dtype0
"
pads

    
¸
%detector/cspdarknet-53/Conv_86/Conv2DConv"detector/cspdarknet-53/Conv_37/mul"
shape
˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
pads

        "
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
strides
"
paddingSAME"
data_formatNHWC"
dtype0

Ž
%detector/cspdarknet-53/Conv_38/Conv2DConvdetector/cspdarknet-53/Pad_3"
strides
"
shape
˙˙˙˙˙˙˙˙˙BB"
dtype0
"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
data_formatNHWC"
paddingVALID"
auto_padVALID
î
9detector/cspdarknet-53/Conv_86/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_86/Conv2D"
data_formatNHWC"
dtype0
"
scale("

bias("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
î
9detector/cspdarknet-53/Conv_38/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_38/Conv2D"
dtype0
"
epsilon%đ'7"

bias("
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC
Ě
(detector/cspdarknet-53/Conv_86/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_86/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
Ő
'detector/cspdarknet-53/Conv_38/SoftplusSoftplus9detector/cspdarknet-53/Conv_38/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_38/TanhTanh'detector/cspdarknet-53/Conv_38/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_38/mulMul9detector/cspdarknet-53/Conv_38/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_38/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¸
%detector/cspdarknet-53/Conv_39/Conv2DConv"detector/cspdarknet-53/Conv_38/mul"
kernel_shape

"
pads

        "
shape
˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
strides
"
paddingSAME"
auto_pad
SAME_LOWER"
dtype0

¸
%detector/cspdarknet-53/Conv_40/Conv2DConv"detector/cspdarknet-53/Conv_38/mul"
strides
"
kernel_shape

"
auto_pad
SAME_LOWER"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME"
data_formatNHWC"
dtype0

î
9detector/cspdarknet-53/Conv_39/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_39/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
data_formatNHWC"
dtype0
"
scale("

bias(
î
9detector/cspdarknet-53/Conv_40/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_40/Conv2D"
data_formatNHWC"
epsilon%đ'7"
dtype0
"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
scale(
Ő
'detector/cspdarknet-53/Conv_39/SoftplusSoftplus9detector/cspdarknet-53/Conv_39/BatchNorm/FusedBatchNormV3"	
dim"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
Ő
'detector/cspdarknet-53/Conv_40/SoftplusSoftplus9detector/cspdarknet-53/Conv_40/BatchNorm/FusedBatchNormV3"	
dim"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_39/TanhTanh'detector/cspdarknet-53/Conv_39/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_40/TanhTanh'detector/cspdarknet-53/Conv_40/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_39/mulMul9detector/cspdarknet-53/Conv_39/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_39/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_40/mulMul9detector/cspdarknet-53/Conv_40/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_40/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

¸
%detector/cspdarknet-53/Conv_41/Conv2DConv"detector/cspdarknet-53/Conv_40/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
paddingSAME"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
pads

        
î
9detector/cspdarknet-53/Conv_41/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_41/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
scale("

bias("
epsilon%đ'7"
dtype0

Ő
'detector/cspdarknet-53/Conv_41/SoftplusSoftplus9detector/cspdarknet-53/Conv_41/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

°
#detector/cspdarknet-53/Conv_41/TanhTanh'detector/cspdarknet-53/Conv_41/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_41/mulMul9detector/cspdarknet-53/Conv_41/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_41/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¸
%detector/cspdarknet-53/Conv_42/Conv2DConv"detector/cspdarknet-53/Conv_41/mul"
shape
˙˙˙˙˙˙˙˙˙  "
strides
"
data_formatNHWC"
pads

    "
kernel_shape

"
dtype0
"
auto_pad
SAME_LOWER"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
î
9detector/cspdarknet-53/Conv_42/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_42/Conv2D"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
data_formatNHWC"
scale("

bias(
Ő
'detector/cspdarknet-53/Conv_42/SoftplusSoftplus9detector/cspdarknet-53/Conv_42/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "	
dim
°
#detector/cspdarknet-53/Conv_42/TanhTanh'detector/cspdarknet-53/Conv_42/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_42/mulMul9detector/cspdarknet-53/Conv_42/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_42/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ť
detector/cspdarknet-53/add_11Add"detector/cspdarknet-53/Conv_40/mul"detector/cspdarknet-53/Conv_42/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ł
%detector/cspdarknet-53/Conv_43/Conv2DConvdetector/cspdarknet-53/add_11"
strides
"
shape
˙˙˙˙˙˙˙˙˙  "
pads

        "
paddingSAME"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
data_formatNHWC"
dtype0

î
9detector/cspdarknet-53/Conv_43/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_43/Conv2D"
data_formatNHWC"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"

bias("
scale(
Ő
'detector/cspdarknet-53/Conv_43/SoftplusSoftplus9detector/cspdarknet-53/Conv_43/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_43/TanhTanh'detector/cspdarknet-53/Conv_43/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_43/mulMul9detector/cspdarknet-53/Conv_43/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_43/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

¸
%detector/cspdarknet-53/Conv_44/Conv2DConv"detector/cspdarknet-53/Conv_43/mul"
dtype0
"
data_formatNHWC"
kernel_shape

"
pads

    "
auto_pad
SAME_LOWER"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME
î
9detector/cspdarknet-53/Conv_44/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_44/Conv2D"
data_formatNHWC"
dtype0
"
epsilon%đ'7"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
scale(
Ő
'detector/cspdarknet-53/Conv_44/SoftplusSoftplus9detector/cspdarknet-53/Conv_44/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim
°
#detector/cspdarknet-53/Conv_44/TanhTanh'detector/cspdarknet-53/Conv_44/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_44/mulMul9detector/cspdarknet-53/Conv_44/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_44/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ś
detector/cspdarknet-53/add_12Adddetector/cspdarknet-53/add_11"detector/cspdarknet-53/Conv_44/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ł
%detector/cspdarknet-53/Conv_45/Conv2DConvdetector/cspdarknet-53/add_12"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
data_formatNHWC"
strides
"
kernel_shape

"
pads

        "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"
paddingSAME
î
9detector/cspdarknet-53/Conv_45/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_45/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
epsilon%đ'7"

bias("
data_formatNHWC"
scale(
Ő
'detector/cspdarknet-53/Conv_45/SoftplusSoftplus9detector/cspdarknet-53/Conv_45/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_45/TanhTanh'detector/cspdarknet-53/Conv_45/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_45/mulMul9detector/cspdarknet-53/Conv_45/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_45/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¸
%detector/cspdarknet-53/Conv_46/Conv2DConv"detector/cspdarknet-53/Conv_45/mul"
kernel_shape

"
pads

    "
shape
˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
dtype0
"
strides

î
9detector/cspdarknet-53/Conv_46/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_46/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"

bias("
scale("
epsilon%đ'7"
data_formatNHWC
Ő
'detector/cspdarknet-53/Conv_46/SoftplusSoftplus9detector/cspdarknet-53/Conv_46/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "	
dim"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_46/TanhTanh'detector/cspdarknet-53/Conv_46/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_46/mulMul9detector/cspdarknet-53/Conv_46/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_46/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Ś
detector/cspdarknet-53/add_13Adddetector/cspdarknet-53/add_12"detector/cspdarknet-53/Conv_46/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ł
%detector/cspdarknet-53/Conv_47/Conv2DConvdetector/cspdarknet-53/add_13"
pads

        "
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
strides
"
data_formatNHWC"
dtype0
"
auto_pad
SAME_LOWER"
paddingSAME"
kernel_shape


î
9detector/cspdarknet-53/Conv_47/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_47/Conv2D"
dtype0
"
epsilon%đ'7"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"

bias(
Ő
'detector/cspdarknet-53/Conv_47/SoftplusSoftplus9detector/cspdarknet-53/Conv_47/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_47/TanhTanh'detector/cspdarknet-53/Conv_47/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_47/mulMul9detector/cspdarknet-53/Conv_47/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_47/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

¸
%detector/cspdarknet-53/Conv_48/Conv2DConv"detector/cspdarknet-53/Conv_47/mul"
pads

    "
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
auto_pad
SAME_LOWER"
strides
"
dtype0
"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  
î
9detector/cspdarknet-53/Conv_48/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_48/Conv2D"
epsilon%đ'7"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"

bias("
scale(
Ő
'detector/cspdarknet-53/Conv_48/SoftplusSoftplus9detector/cspdarknet-53/Conv_48/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "	
dim"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_48/TanhTanh'detector/cspdarknet-53/Conv_48/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_48/mulMul9detector/cspdarknet-53/Conv_48/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_48/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ś
detector/cspdarknet-53/add_14Adddetector/cspdarknet-53/add_13"detector/cspdarknet-53/Conv_48/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ł
%detector/cspdarknet-53/Conv_49/Conv2DConvdetector/cspdarknet-53/add_14"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
kernel_shape

"
data_formatNHWC"
strides
"
pads

        
î
9detector/cspdarknet-53/Conv_49/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_49/Conv2D"
dtype0
"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "

bias("
scale("
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_49/SoftplusSoftplus9detector/cspdarknet-53/Conv_49/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_49/TanhTanh'detector/cspdarknet-53/Conv_49/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_49/mulMul9detector/cspdarknet-53/Conv_49/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_49/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¸
%detector/cspdarknet-53/Conv_50/Conv2DConv"detector/cspdarknet-53/Conv_49/mul"
dtype0
"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
paddingSAME"
pads

    "
auto_pad
SAME_LOWER"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  
î
9detector/cspdarknet-53/Conv_50/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_50/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
data_formatNHWC"

bias("
scale("
dtype0

Ő
'detector/cspdarknet-53/Conv_50/SoftplusSoftplus9detector/cspdarknet-53/Conv_50/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim
°
#detector/cspdarknet-53/Conv_50/TanhTanh'detector/cspdarknet-53/Conv_50/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_50/mulMul9detector/cspdarknet-53/Conv_50/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_50/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ś
detector/cspdarknet-53/add_15Adddetector/cspdarknet-53/add_14"detector/cspdarknet-53/Conv_50/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ł
%detector/cspdarknet-53/Conv_51/Conv2DConvdetector/cspdarknet-53/add_15"
paddingSAME"
auto_pad
SAME_LOWER"
strides
"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
pads

        "
kernel_shape

"
data_formatNHWC
î
9detector/cspdarknet-53/Conv_51/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_51/Conv2D"
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
data_formatNHWC"

bias(
Ő
'detector/cspdarknet-53/Conv_51/SoftplusSoftplus9detector/cspdarknet-53/Conv_51/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "	
dim
°
#detector/cspdarknet-53/Conv_51/TanhTanh'detector/cspdarknet-53/Conv_51/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_51/mulMul9detector/cspdarknet-53/Conv_51/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_51/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¸
%detector/cspdarknet-53/Conv_52/Conv2DConv"detector/cspdarknet-53/Conv_51/mul"
dtype0
"
strides
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
data_formatNHWC"
pads

    "
auto_pad
SAME_LOWER"
paddingSAME
î
9detector/cspdarknet-53/Conv_52/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_52/Conv2D"
dtype0
"

bias("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
scale(
Ő
'detector/cspdarknet-53/Conv_52/SoftplusSoftplus9detector/cspdarknet-53/Conv_52/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_52/TanhTanh'detector/cspdarknet-53/Conv_52/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_52/mulMul9detector/cspdarknet-53/Conv_52/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_52/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ś
detector/cspdarknet-53/add_16Adddetector/cspdarknet-53/add_15"detector/cspdarknet-53/Conv_52/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ł
%detector/cspdarknet-53/Conv_53/Conv2DConvdetector/cspdarknet-53/add_16"
paddingSAME"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
data_formatNHWC"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"
pads

        
î
9detector/cspdarknet-53/Conv_53/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_53/Conv2D"
epsilon%đ'7"
dtype0
"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"

bias(
Ő
'detector/cspdarknet-53/Conv_53/SoftplusSoftplus9detector/cspdarknet-53/Conv_53/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0

°
#detector/cspdarknet-53/Conv_53/TanhTanh'detector/cspdarknet-53/Conv_53/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_53/mulMul9detector/cspdarknet-53/Conv_53/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_53/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

¸
%detector/cspdarknet-53/Conv_54/Conv2DConv"detector/cspdarknet-53/Conv_53/mul"
paddingSAME"
dtype0
"
pads

    "
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
shape
˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
strides

î
9detector/cspdarknet-53/Conv_54/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_54/Conv2D"
dtype0
"
scale("
epsilon%đ'7"
data_formatNHWC"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ő
'detector/cspdarknet-53/Conv_54/SoftplusSoftplus9detector/cspdarknet-53/Conv_54/BatchNorm/FusedBatchNormV3"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0

°
#detector/cspdarknet-53/Conv_54/TanhTanh'detector/cspdarknet-53/Conv_54/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_54/mulMul9detector/cspdarknet-53/Conv_54/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_54/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Ś
detector/cspdarknet-53/add_17Adddetector/cspdarknet-53/add_16"detector/cspdarknet-53/Conv_54/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ł
%detector/cspdarknet-53/Conv_55/Conv2DConvdetector/cspdarknet-53/add_17"
data_formatNHWC"
pads

        "
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  "
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
î
9detector/cspdarknet-53/Conv_55/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_55/Conv2D"

bias("
scale("
dtype0
"
data_formatNHWC"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ő
'detector/cspdarknet-53/Conv_55/SoftplusSoftplus9detector/cspdarknet-53/Conv_55/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim
°
#detector/cspdarknet-53/Conv_55/TanhTanh'detector/cspdarknet-53/Conv_55/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_55/mulMul9detector/cspdarknet-53/Conv_55/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_55/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

¸
%detector/cspdarknet-53/Conv_56/Conv2DConv"detector/cspdarknet-53/Conv_55/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
pads

    "
paddingSAME"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
auto_pad
SAME_LOWER"
strides

î
9detector/cspdarknet-53/Conv_56/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_56/Conv2D"

bias("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
scale("
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_56/SoftplusSoftplus9detector/cspdarknet-53/Conv_56/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_56/TanhTanh'detector/cspdarknet-53/Conv_56/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_56/mulMul9detector/cspdarknet-53/Conv_56/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_56/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Ś
detector/cspdarknet-53/add_18Adddetector/cspdarknet-53/add_17"detector/cspdarknet-53/Conv_56/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ł
%detector/cspdarknet-53/Conv_57/Conv2DConvdetector/cspdarknet-53/add_18"
strides
"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
paddingSAME"
data_formatNHWC"
auto_pad
SAME_LOWER"
pads

        
î
9detector/cspdarknet-53/Conv_57/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_57/Conv2D"
dtype0
"

bias("
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC
Ő
'detector/cspdarknet-53/Conv_57/SoftplusSoftplus9detector/cspdarknet-53/Conv_57/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"	
dim
°
#detector/cspdarknet-53/Conv_57/TanhTanh'detector/cspdarknet-53/Conv_57/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Č
"detector/cspdarknet-53/Conv_57/mulMul9detector/cspdarknet-53/Conv_57/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_57/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ź
detector/cspdarknet-53/concat_3Concat"detector/cspdarknet-53/Conv_57/mul"detector/cspdarknet-53/Conv_39/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "

axis"
dtype0

ľ
%detector/cspdarknet-53/Conv_58/Conv2DConvdetector/cspdarknet-53/concat_3"
auto_pad
SAME_LOWER"
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
strides
"
pads

        
î
9detector/cspdarknet-53/Conv_58/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_58/Conv2D"

bias("
scale("
epsilon%đ'7"
dtype0
"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ő
'detector/cspdarknet-53/Conv_58/SoftplusSoftplus9detector/cspdarknet-53/Conv_58/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
#detector/cspdarknet-53/Conv_58/TanhTanh'detector/cspdarknet-53/Conv_58/Softplus"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Č
"detector/cspdarknet-53/Conv_58/mulMul9detector/cspdarknet-53/Conv_58/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_58/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
detector/cspdarknet-53/Pad_4Pad"detector/cspdarknet-53/Conv_58/mul"
mode
constant"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"""
dtype0
"
pads

    
¸
%detector/cspdarknet-53/Conv_79/Conv2DConv"detector/cspdarknet-53/Conv_58/mul"
shape
˙˙˙˙˙˙˙˙˙  "
pads

        "
kernel_shape

"
paddingSAME"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
strides

Ž
%detector/cspdarknet-53/Conv_59/Conv2DConvdetector/cspdarknet-53/Pad_4"
pads

        "
data_formatNHWC"
paddingVALID"
shape
˙˙˙˙˙˙˙˙˙"""
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
auto_padVALID"
dtype0
"
kernel_shape


î
9detector/cspdarknet-53/Conv_79/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_79/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
dtype0
"

bias("
data_formatNHWC"
scale(
î
9detector/cspdarknet-53/Conv_59/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_59/Conv2D"

bias("
scale("
data_formatNHWC"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
epsilon%đ'7
Ě
(detector/cspdarknet-53/Conv_79/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_79/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
Ő
'detector/cspdarknet-53/Conv_59/SoftplusSoftplus9detector/cspdarknet-53/Conv_59/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_59/TanhTanh'detector/cspdarknet-53/Conv_59/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_59/mulMul9detector/cspdarknet-53/Conv_59/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_59/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
%detector/cspdarknet-53/Conv_60/Conv2DConv"detector/cspdarknet-53/Conv_59/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER"
strides
"
pads

        "
paddingSAME"
data_formatNHWC"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape


¸
%detector/cspdarknet-53/Conv_61/Conv2DConv"detector/cspdarknet-53/Conv_59/mul"
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
pads

        "
data_formatNHWC"
auto_pad
SAME_LOWER"
paddingSAME"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"
strides

î
9detector/cspdarknet-53/Conv_60/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_60/Conv2D"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
dtype0
"

bias("
scale(
î
9detector/cspdarknet-53/Conv_61/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_61/Conv2D"
data_formatNHWC"
epsilon%đ'7"

bias("
scale("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
'detector/cspdarknet-53/Conv_60/SoftplusSoftplus9detector/cspdarknet-53/Conv_60/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"	
dim
Ő
'detector/cspdarknet-53/Conv_61/SoftplusSoftplus9detector/cspdarknet-53/Conv_61/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_60/TanhTanh'detector/cspdarknet-53/Conv_60/Softplus"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_61/TanhTanh'detector/cspdarknet-53/Conv_61/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

Č
"detector/cspdarknet-53/Conv_60/mulMul9detector/cspdarknet-53/Conv_60/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_60/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_61/mulMul9detector/cspdarknet-53/Conv_61/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_61/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

¸
%detector/cspdarknet-53/Conv_62/Conv2DConv"detector/cspdarknet-53/Conv_61/mul"
paddingSAME"
pads

        "
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
data_formatNHWC"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_62/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_62/Conv2D"
epsilon%đ'7"

bias("
data_formatNHWC"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
scale(
Ő
'detector/cspdarknet-53/Conv_62/SoftplusSoftplus9detector/cspdarknet-53/Conv_62/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_62/TanhTanh'detector/cspdarknet-53/Conv_62/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_62/mulMul9detector/cspdarknet-53/Conv_62/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_62/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

¸
%detector/cspdarknet-53/Conv_63/Conv2DConv"detector/cspdarknet-53/Conv_62/mul"
paddingSAME"
data_formatNHWC"
strides
"
kernel_shape

"
dtype0
"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_63/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_63/Conv2D"
data_formatNHWC"

bias("
scale("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_63/SoftplusSoftplus9detector/cspdarknet-53/Conv_63/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"	
dim"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

°
#detector/cspdarknet-53/Conv_63/TanhTanh'detector/cspdarknet-53/Conv_63/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

Č
"detector/cspdarknet-53/Conv_63/mulMul9detector/cspdarknet-53/Conv_63/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_63/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
detector/cspdarknet-53/add_19Add"detector/cspdarknet-53/Conv_61/mul"detector/cspdarknet-53/Conv_63/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ł
%detector/cspdarknet-53/Conv_64/Conv2DConvdetector/cspdarknet-53/add_19"
strides
"
pads

        "
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
9detector/cspdarknet-53/Conv_64/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_64/Conv2D"
data_formatNHWC"
scale("

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
epsilon%đ'7"
dtype0

Ő
'detector/cspdarknet-53/Conv_64/SoftplusSoftplus9detector/cspdarknet-53/Conv_64/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"	
dim"
dtype0

°
#detector/cspdarknet-53/Conv_64/TanhTanh'detector/cspdarknet-53/Conv_64/Softplus"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_64/mulMul9detector/cspdarknet-53/Conv_64/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_64/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

¸
%detector/cspdarknet-53/Conv_65/Conv2DConv"detector/cspdarknet-53/Conv_64/mul"
paddingSAME"
kernel_shape

"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
pads

    "
strides
"
shape
˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_65/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_65/Conv2D"
epsilon%đ'7"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"

bias("
dtype0

Ő
'detector/cspdarknet-53/Conv_65/SoftplusSoftplus9detector/cspdarknet-53/Conv_65/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"	
dim
°
#detector/cspdarknet-53/Conv_65/TanhTanh'detector/cspdarknet-53/Conv_65/Softplus"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

Č
"detector/cspdarknet-53/Conv_65/mulMul9detector/cspdarknet-53/Conv_65/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_65/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
detector/cspdarknet-53/add_20Adddetector/cspdarknet-53/add_19"detector/cspdarknet-53/Conv_65/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ł
%detector/cspdarknet-53/Conv_66/Conv2DConvdetector/cspdarknet-53/add_20"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
pads

        "
strides
"
kernel_shape

"
auto_pad
SAME_LOWER"
data_formatNHWC"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙
î
9detector/cspdarknet-53/Conv_66/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_66/Conv2D"
epsilon%đ'7"
data_formatNHWC"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
scale(
Ő
'detector/cspdarknet-53/Conv_66/SoftplusSoftplus9detector/cspdarknet-53/Conv_66/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"	
dim"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_66/TanhTanh'detector/cspdarknet-53/Conv_66/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_66/mulMul9detector/cspdarknet-53/Conv_66/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_66/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
%detector/cspdarknet-53/Conv_67/Conv2DConv"detector/cspdarknet-53/Conv_66/mul"
shape
˙˙˙˙˙˙˙˙˙"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
kernel_shape

"
paddingSAME"
dtype0

î
9detector/cspdarknet-53/Conv_67/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_67/Conv2D"
epsilon%đ'7"
data_formatNHWC"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"

bias(
Ő
'detector/cspdarknet-53/Conv_67/SoftplusSoftplus9detector/cspdarknet-53/Conv_67/BatchNorm/FusedBatchNormV3"	
dim"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_67/TanhTanh'detector/cspdarknet-53/Conv_67/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_67/mulMul9detector/cspdarknet-53/Conv_67/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_67/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

Ś
detector/cspdarknet-53/add_21Adddetector/cspdarknet-53/add_20"detector/cspdarknet-53/Conv_67/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ł
%detector/cspdarknet-53/Conv_68/Conv2DConvdetector/cspdarknet-53/add_21"
shape
˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
paddingSAME"
strides
"
dtype0
"
kernel_shape

"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
pads

        
î
9detector/cspdarknet-53/Conv_68/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_68/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"

bias("
dtype0
"
epsilon%đ'7"
scale(
Ő
'detector/cspdarknet-53/Conv_68/SoftplusSoftplus9detector/cspdarknet-53/Conv_68/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"	
dim
°
#detector/cspdarknet-53/Conv_68/TanhTanh'detector/cspdarknet-53/Conv_68/Softplus"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_68/mulMul9detector/cspdarknet-53/Conv_68/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_68/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

¸
%detector/cspdarknet-53/Conv_69/Conv2DConv"detector/cspdarknet-53/Conv_68/mul"
strides
"
shape
˙˙˙˙˙˙˙˙˙"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
data_formatNHWC"
pads

    "
auto_pad
SAME_LOWER"
kernel_shape


î
9detector/cspdarknet-53/Conv_69/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_69/Conv2D"
epsilon%đ'7"
data_formatNHWC"

bias("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
scale(
Ő
'detector/cspdarknet-53/Conv_69/SoftplusSoftplus9detector/cspdarknet-53/Conv_69/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"	
dim
°
#detector/cspdarknet-53/Conv_69/TanhTanh'detector/cspdarknet-53/Conv_69/Softplus"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
Č
"detector/cspdarknet-53/Conv_69/mulMul9detector/cspdarknet-53/Conv_69/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_69/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

Ś
detector/cspdarknet-53/add_22Adddetector/cspdarknet-53/add_21"detector/cspdarknet-53/Conv_69/mul"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ł
%detector/cspdarknet-53/Conv_70/Conv2DConvdetector/cspdarknet-53/add_22"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
strides
"
kernel_shape

"
pads

        "
paddingSAME"
dtype0

î
9detector/cspdarknet-53/Conv_70/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_70/Conv2D"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
scale("

bias("
data_formatNHWC"
dtype0

Ő
'detector/cspdarknet-53/Conv_70/SoftplusSoftplus9detector/cspdarknet-53/Conv_70/BatchNorm/FusedBatchNormV3"
dtype0
"	
dim"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_70/TanhTanh'detector/cspdarknet-53/Conv_70/Softplus"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

Č
"detector/cspdarknet-53/Conv_70/mulMul9detector/cspdarknet-53/Conv_70/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_70/Tanh"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
detector/cspdarknet-53/concat_4Concat"detector/cspdarknet-53/Conv_70/mul"detector/cspdarknet-53/Conv_60/mul"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"

axis
ľ
%detector/cspdarknet-53/Conv_71/Conv2DConvdetector/cspdarknet-53/concat_4"
dtype0
"
pads

        "
shape
˙˙˙˙˙˙˙˙˙"
strides
"
paddingSAME"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
kernel_shape

"
data_formatNHWC
î
9detector/cspdarknet-53/Conv_71/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_71/Conv2D"
scale("
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"

bias("
epsilon%đ'7
Ő
'detector/cspdarknet-53/Conv_71/SoftplusSoftplus9detector/cspdarknet-53/Conv_71/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"	
dim"
shape
˙˙˙˙˙˙˙˙˙
°
#detector/cspdarknet-53/Conv_71/TanhTanh'detector/cspdarknet-53/Conv_71/Softplus"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

Č
"detector/cspdarknet-53/Conv_71/mulMul9detector/cspdarknet-53/Conv_71/BatchNorm/FusedBatchNormV3#detector/cspdarknet-53/Conv_71/Tanh"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

¸
%detector/cspdarknet-53/Conv_72/Conv2DConv"detector/cspdarknet-53/Conv_71/mul"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER"
dtype0
"
data_formatNHWC"
strides
"
kernel_shape

"
paddingSAME
î
9detector/cspdarknet-53/Conv_72/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_72/Conv2D"
epsilon%đ'7"

bias("
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
data_formatNHWC
Ě
(detector/cspdarknet-53/Conv_72/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_72/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ž
%detector/cspdarknet-53/Conv_73/Conv2DConv(detector/cspdarknet-53/Conv_72/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
strides
"
paddingSAME"
pads

    "
auto_pad
SAME_LOWER"
data_formatNHWC"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape


î
9detector/cspdarknet-53/Conv_73/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_73/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
epsilon%đ'7"
data_formatNHWC"
scale("

bias(
Ě
(detector/cspdarknet-53/Conv_73/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_73/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

ž
%detector/cspdarknet-53/Conv_74/Conv2DConv(detector/cspdarknet-53/Conv_73/LeakyRelu"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙"
pads

        "
data_formatNHWC"
auto_pad
SAME_LOWER"
dtype0
"
strides
"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
9detector/cspdarknet-53/Conv_74/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_74/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"

bias("
dtype0
"
data_formatNHWC"
scale("
epsilon%đ'7
Ě
(detector/cspdarknet-53/Conv_74/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_74/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙
Ś
(detector/cspdarknet-53/MaxPool2D/MaxPoolPool(detector/cspdarknet-53/Conv_74/LeakyRelu"
pooling_typeMAX"
auto_pad
SAME_LOWER"
strides
"
data_formatNHWC"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
pads

    "
kernel_shape

¨
*detector/cspdarknet-53/MaxPool2D_1/MaxPoolPool(detector/cspdarknet-53/Conv_74/LeakyRelu"
data_formatNHWC"
dtype0
"
pads

    "
auto_pad
SAME_LOWER"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
pooling_typeMAX"
kernel_shape
		
¨
*detector/cspdarknet-53/MaxPool2D_2/MaxPoolPool(detector/cspdarknet-53/Conv_74/LeakyRelu"
data_formatNHWC"
auto_pad
SAME_LOWER"
pads

    "
strides
"
dtype0
"
kernel_shape
"
pooling_typeMAX"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
detector/cspdarknet-53/concat_5Concat(detector/cspdarknet-53/MaxPool2D/MaxPool*detector/cspdarknet-53/MaxPool2D_1/MaxPool*detector/cspdarknet-53/MaxPool2D_2/MaxPool(detector/cspdarknet-53/Conv_74/LeakyRelu"

axis"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

ľ
%detector/cspdarknet-53/Conv_75/Conv2DConvdetector/cspdarknet-53/concat_5"
paddingSAME"
strides
"
auto_pad
SAME_LOWER"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
shape
˙˙˙˙˙˙˙˙˙"
pads

        "
kernel_shape


î
9detector/cspdarknet-53/Conv_75/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_75/Conv2D"
scale("

bias("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
epsilon%đ'7
Ě
(detector/cspdarknet-53/Conv_75/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_75/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
%detector/cspdarknet-53/Conv_76/Conv2DConv(detector/cspdarknet-53/Conv_75/LeakyRelu"
paddingSAME"
auto_pad
SAME_LOWER"
strides
"
dtype0
"
pads

    "
kernel_shape

"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙
î
9detector/cspdarknet-53/Conv_76/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_76/Conv2D"
epsilon%đ'7"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
scale("

bias(
Ě
(detector/cspdarknet-53/Conv_76/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_76/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙
ž
%detector/cspdarknet-53/Conv_77/Conv2DConv(detector/cspdarknet-53/Conv_76/LeakyRelu"
data_formatNHWC"
dtype0
"
auto_pad
SAME_LOWER"
pads

        "
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
strides
"
kernel_shape

"
paddingSAME
î
9detector/cspdarknet-53/Conv_77/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_77/Conv2D"
scale("
dtype0
"

bias("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC
Ě
(detector/cspdarknet-53/Conv_77/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_77/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

ž
%detector/cspdarknet-53/Conv_78/Conv2DConv(detector/cspdarknet-53/Conv_77/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
dtype0
"
auto_pad
SAME_LOWER"
pads

        "
shape
˙˙˙˙˙˙˙˙˙"
paddingSAME"
strides
"
kernel_shape


î
9detector/cspdarknet-53/Conv_78/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_78/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
epsilon%đ'7"
dtype0
"
scale("

bias("
data_formatNHWC
Ě
(detector/cspdarknet-53/Conv_78/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_78/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙

,detector/cspdarknet-53/ResizeNearestNeighborResizeNearestNeighbor(detector/cspdarknet-53/Conv_78/LeakyRelu1detector/cspdarknet-53/ResizeNearestNeighbor/size"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
size
"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

Ě
detector/cspdarknet-53/concat_6Concat(detector/cspdarknet-53/Conv_79/LeakyRelu,detector/cspdarknet-53/ResizeNearestNeighbor"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "

axis
ľ
%detector/cspdarknet-53/Conv_80/Conv2DConvdetector/cspdarknet-53/concat_6"
pads

        "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
paddingSAME"
data_formatNHWC"
shape
˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
strides
"
kernel_shape


î
9detector/cspdarknet-53/Conv_80/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_80/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
scale("
data_formatNHWC"
dtype0
"

bias(
Ě
(detector/cspdarknet-53/Conv_80/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_80/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
ž
%detector/cspdarknet-53/Conv_81/Conv2DConv(detector/cspdarknet-53/Conv_80/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
strides
"
paddingSAME"
auto_pad
SAME_LOWER"
data_formatNHWC"
pads

    "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
î
9detector/cspdarknet-53/Conv_81/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_81/Conv2D"
data_formatNHWC"

bias("
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

Ě
(detector/cspdarknet-53/Conv_81/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_81/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ž
%detector/cspdarknet-53/Conv_82/Conv2DConv(detector/cspdarknet-53/Conv_81/LeakyRelu"
dtype0
"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
paddingSAME"
pads

        "
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙  "
strides

î
9detector/cspdarknet-53/Conv_82/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_82/Conv2D"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"

bias("
scale("
epsilon%đ'7
Ě
(detector/cspdarknet-53/Conv_82/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_82/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ž
%detector/cspdarknet-53/Conv_83/Conv2DConv(detector/cspdarknet-53/Conv_82/LeakyRelu"
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
strides
"
pads

    "
data_formatNHWC"
kernel_shape

"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_83/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_83/Conv2D"
epsilon%đ'7"
scale("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"

bias(
Ě
(detector/cspdarknet-53/Conv_83/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_83/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
ž
%detector/cspdarknet-53/Conv_84/Conv2DConv(detector/cspdarknet-53/Conv_83/LeakyRelu"
strides
"
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
pads

        "
data_formatNHWC"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME
î
9detector/cspdarknet-53/Conv_84/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_84/Conv2D"
scale("
epsilon%đ'7"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"

bias(
Ě
(detector/cspdarknet-53/Conv_84/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_84/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
ž
%detector/cspdarknet-53/Conv_85/Conv2DConv(detector/cspdarknet-53/Conv_84/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
pads

        "
dtype0
"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙  "
strides
"
kernel_shape

"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_85/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_85/Conv2D"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
scale("
epsilon%đ'7"

bias("
dtype0

Ě
(detector/cspdarknet-53/Conv_85/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_85/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  "
dtype0


.detector/cspdarknet-53/ResizeNearestNeighbor_1ResizeNearestNeighbor(detector/cspdarknet-53/Conv_85/LeakyRelu3detector/cspdarknet-53/ResizeNearestNeighbor_1/size"
size
"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Î
detector/cspdarknet-53/concat_7Concat(detector/cspdarknet-53/Conv_86/LeakyRelu.detector/cspdarknet-53/ResizeNearestNeighbor_1"

axis"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

ľ
%detector/cspdarknet-53/Conv_87/Conv2DConvdetector/cspdarknet-53/concat_7"
paddingSAME"
auto_pad
SAME_LOWER"
data_formatNHWC"
pads

        "
strides
"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
dtype0

î
9detector/cspdarknet-53/Conv_87/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_87/Conv2D"
data_formatNHWC"

bias("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
scale("
dtype0

Ě
(detector/cspdarknet-53/Conv_87/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_87/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
ž
%detector/cspdarknet-53/Conv_88/Conv2DConv(detector/cspdarknet-53/Conv_87/LeakyRelu"
data_formatNHWC"
dtype0
"
strides
"
paddingSAME"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_88/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_88/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
epsilon%đ'7"
scale("

bias("
data_formatNHWC"
dtype0

Ě
(detector/cspdarknet-53/Conv_88/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_88/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

ž
%detector/cspdarknet-53/Conv_89/Conv2DConv(detector/cspdarknet-53/Conv_88/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
dtype0
"
strides
"
shape
˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
kernel_shape

"
paddingSAME"
pads

        
î
9detector/cspdarknet-53/Conv_89/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_89/Conv2D"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
dtype0
"
epsilon%đ'7"
scale(
Ě
(detector/cspdarknet-53/Conv_89/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_89/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
shape
˙˙˙˙˙˙˙˙˙@@
ž
%detector/cspdarknet-53/Conv_90/Conv2DConv(detector/cspdarknet-53/Conv_89/LeakyRelu"
paddingSAME"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
kernel_shape

"
data_formatNHWC"
dtype0
"
pads

    "
shape
˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER
î
9detector/cspdarknet-53/Conv_90/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_90/Conv2D"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
data_formatNHWC"
scale("

bias("
epsilon%đ'7
Ě
(detector/cspdarknet-53/Conv_90/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_90/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

ž
%detector/cspdarknet-53/Conv_91/Conv2DConv(detector/cspdarknet-53/Conv_90/LeakyRelu"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"
pads

        "
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
strides
"
data_formatNHWC"
auto_pad
SAME_LOWER"
paddingSAME
î
9detector/cspdarknet-53/Conv_91/BatchNorm/FusedBatchNormV3	BatchNorm%detector/cspdarknet-53/Conv_91/Conv2D"
data_formatNHWC"
epsilon%đ'7"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
scale(
Ě
(detector/cspdarknet-53/Conv_91/LeakyRelu	LeakyRelu9detector/cspdarknet-53/Conv_91/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙@@"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0

ľ
detector/yolo-v4/Conv/Conv2DConv(detector/cspdarknet-53/Conv_91/LeakyRelu"
paddingSAME"
kernel_shape

"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
strides
"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@"
auto_pad
SAME_LOWER"
data_formatNHWC
Ž
detector/yolo-v4/PadPad(detector/cspdarknet-53/Conv_91/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙BB"
dtype0
"
pads

    "
mode
constant
Ü
0detector/yolo-v4/Conv/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"

bias("
data_formatNHWC"
epsilon%đ'7"
scale(

detector/yolo-v4/Conv_2/Conv2DConvdetector/yolo-v4/Pad"
paddingVALID"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙BB"
kernel_shape

"
data_formatNHWC"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
auto_padVALID"
pads

        
ş
detector/yolo-v4/Conv/LeakyRelu	LeakyRelu0detector/yolo-v4/Conv/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙@@
ŕ
2detector/yolo-v4/Conv_2/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_2/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
epsilon%đ'7"

bias("
dtype0
"
scale(
ž
detector/yolo-v4/Conv_1/Conv2DConvdetector/yolo-v4/Conv/LeakyRelu"
shape
˙˙˙˙˙˙˙˙˙@@"
kernel_shape

˙"
auto_pad
SAME_LOWER"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙@@˙"
data_formatNHWC"
dtype0
"
paddingSAME"
pads

        "
use_bias(
ž
!detector/yolo-v4/Conv_2/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_2/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ť
detector/yolo-v4/concat_3Concat!detector/yolo-v4/Conv_2/LeakyRelu(detector/cspdarknet-53/Conv_84/LeakyRelu"
dtype0
"

axis"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
¨
detector/yolo-v4/Conv_3/Conv2DConvdetector/yolo-v4/concat_3"
kernel_shape

"
shape
˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"
data_formatNHWC"
dtype0
"
strides
"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
pads

        
ŕ
2detector/yolo-v4/Conv_3/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_3/Conv2D"
epsilon%đ'7"

bias("
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
scale(
ž
!detector/yolo-v4/Conv_3/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_3/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
°
detector/yolo-v4/Conv_4/Conv2DConv!detector/yolo-v4/Conv_3/LeakyRelu"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
paddingSAME"
data_formatNHWC"
strides
"
dtype0
"
pads

    "
auto_pad
SAME_LOWER
ŕ
2detector/yolo-v4/Conv_4/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_4/Conv2D"
data_formatNHWC"

bias("
epsilon%đ'7"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ž
!detector/yolo-v4/Conv_4/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_4/BatchNorm/FusedBatchNormV3"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
shape
˙˙˙˙˙˙˙˙˙  
°
detector/yolo-v4/Conv_5/Conv2DConv!detector/yolo-v4/Conv_4/LeakyRelu"
strides
"
pads

        "
kernel_shape

"
data_formatNHWC"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙  "
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

ŕ
2detector/yolo-v4/Conv_5/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_5/Conv2D"
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
epsilon%đ'7"
dtype0
"

bias("
data_formatNHWC
ž
!detector/yolo-v4/Conv_5/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_5/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
°
detector/yolo-v4/Conv_6/Conv2DConv!detector/yolo-v4/Conv_5/LeakyRelu"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙  "
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
pads

    "
data_formatNHWC"
strides
"
dtype0
"
kernel_shape


ŕ
2detector/yolo-v4/Conv_6/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_6/Conv2D"

bias("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"
dtype0
"
scale(
ž
!detector/yolo-v4/Conv_6/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_6/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

°
detector/yolo-v4/Conv_7/Conv2DConv!detector/yolo-v4/Conv_6/LeakyRelu"
pads

        "
auto_pad
SAME_LOWER"
kernel_shape

"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
paddingSAME"
strides
"
data_formatNHWC
ŕ
2detector/yolo-v4/Conv_7/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_7/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
data_formatNHWC"

bias("
scale("
dtype0
"
epsilon%đ'7
ž
!detector/yolo-v4/Conv_7/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_7/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙  "0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0

°
detector/yolo-v4/Conv_8/Conv2DConv!detector/yolo-v4/Conv_7/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
pads

    "
paddingSAME"
auto_pad
SAME_LOWER"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  "
kernel_shape

"
data_formatNHWC"
strides

Š
detector/yolo-v4/Pad_1Pad!detector/yolo-v4/Conv_7/LeakyRelu"
dtype0
"
pads

    "0
_output_shapes
:˙˙˙˙˙˙˙˙˙"""
mode
constant
ŕ
2detector/yolo-v4/Conv_8/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_8/Conv2D"
dtype0
"
epsilon%đ'7"
scale("
data_formatNHWC"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙  
˘
detector/yolo-v4/Conv_10/Conv2DConvdetector/yolo-v4/Pad_1"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"""
paddingVALID"
kernel_shape

"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
strides
"
auto_padVALID"
pads

        
ž
!detector/yolo-v4/Conv_8/LeakyRelu	LeakyRelu2detector/yolo-v4/Conv_8/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  "
dtype0
"
shape
˙˙˙˙˙˙˙˙˙  
â
3detector/yolo-v4/Conv_10/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_10/Conv2D"

bias("0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
dtype0
"
scale("
epsilon%đ'7
Ŕ
detector/yolo-v4/Conv_9/Conv2DConv!detector/yolo-v4/Conv_8/LeakyRelu"
pads

        "
auto_pad
SAME_LOWER"
dtype0
"
strides
"
data_formatNHWC"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙  "
use_bias("
kernel_shape

˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙  ˙
Ŕ
"detector/yolo-v4/Conv_10/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_10/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

ź
detector/yolo-v4/concat_7Concat"detector/yolo-v4/Conv_10/LeakyRelu(detector/cspdarknet-53/Conv_77/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"

axis
Š
detector/yolo-v4/Conv_11/Conv2DConvdetector/yolo-v4/concat_7"
paddingSAME"
dtype0
"
data_formatNHWC"
pads

        "
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape

"
auto_pad
SAME_LOWER"
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
3detector/yolo-v4/Conv_11/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_11/Conv2D"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC"
scale("
epsilon%đ'7"

bias("
dtype0

Ŕ
"detector/yolo-v4/Conv_11/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_11/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"
dtype0
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
detector/yolo-v4/Conv_12/Conv2DConv"detector/yolo-v4/Conv_11/LeakyRelu"
auto_pad
SAME_LOWER"
dtype0
"
data_formatNHWC"
paddingSAME"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
kernel_shape

"
pads

    "
strides

â
3detector/yolo-v4/Conv_12/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_12/Conv2D"
dtype0
"

bias("
data_formatNHWC"
scale("
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
"detector/yolo-v4/Conv_12/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_12/BatchNorm/FusedBatchNormV3"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
detector/yolo-v4/Conv_13/Conv2DConv"detector/yolo-v4/Conv_12/LeakyRelu"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
kernel_shape

"
paddingSAME"
auto_pad
SAME_LOWER"
shape
˙˙˙˙˙˙˙˙˙"
strides
"
dtype0
"
data_formatNHWC"
pads

        
â
3detector/yolo-v4/Conv_13/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_13/Conv2D"
epsilon%đ'7"
data_formatNHWC"
dtype0
"

bias("
scale("0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
"detector/yolo-v4/Conv_13/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_13/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
shape
˙˙˙˙˙˙˙˙˙
˛
detector/yolo-v4/Conv_14/Conv2DConv"detector/yolo-v4/Conv_13/LeakyRelu"
dtype0
"
pads

    "
shape
˙˙˙˙˙˙˙˙˙"
paddingSAME"
data_formatNHWC"
auto_pad
SAME_LOWER"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
strides
"
kernel_shape


â
3detector/yolo-v4/Conv_14/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_14/Conv2D"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"

bias("
dtype0
"
scale("
data_formatNHWC
Ŕ
"detector/yolo-v4/Conv_14/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_14/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

˛
detector/yolo-v4/Conv_15/Conv2DConv"detector/yolo-v4/Conv_14/LeakyRelu"
kernel_shape

"
dtype0
"
auto_pad
SAME_LOWER"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
strides
"
paddingSAME"
pads

        "
shape
˙˙˙˙˙˙˙˙˙
â
3detector/yolo-v4/Conv_15/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_15/Conv2D"

bias("
scale("
dtype0
"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
data_formatNHWC
Ŕ
"detector/yolo-v4/Conv_15/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_15/BatchNorm/FusedBatchNormV3"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
shape
˙˙˙˙˙˙˙˙˙"
dtype0

˛
detector/yolo-v4/Conv_16/Conv2DConv"detector/yolo-v4/Conv_15/LeakyRelu"
shape
˙˙˙˙˙˙˙˙˙"
pads

    "
auto_pad
SAME_LOWER"
strides
"
kernel_shape

"
dtype0
"
paddingSAME"
data_formatNHWC"0
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
3detector/yolo-v4/Conv_16/BatchNorm/FusedBatchNormV3	BatchNormdetector/yolo-v4/Conv_16/Conv2D"
epsilon%đ'7"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0
"
data_formatNHWC"

bias("
scale(
Ŕ
"detector/yolo-v4/Conv_16/LeakyRelu	LeakyRelu3detector/yolo-v4/Conv_16/BatchNorm/FusedBatchNormV3"
shape
˙˙˙˙˙˙˙˙˙"0
_output_shapes
:˙˙˙˙˙˙˙˙˙"
dtype0

Â
detector/yolo-v4/Conv_17/Conv2DConv"detector/yolo-v4/Conv_16/LeakyRelu"
data_formatNHWC"
kernel_shape

˙"
use_bias("
strides
"0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙"
paddingSAME"
shape
˙˙˙˙˙˙˙˙˙"
auto_pad
SAME_LOWER"
dtype0
"
pads

        