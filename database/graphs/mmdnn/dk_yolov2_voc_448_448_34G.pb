
S
layer23-actRelulayer23-conv"0
_output_shapes
:����������
Q

layer6-actRelulayer6-conv"0
_output_shapes
:���������pp�
�
layer5-convConv
layer4-act"
group"
kernel_shape	
�@"
strides
"
use_bias("/
_output_shapes
:���������pp@"
pads

        
�
layer23-convConvlayer22-act"
use_bias("0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

��"
strides

�
layer0-convConvdata"
kernel_shape
 "
strides
"
use_bias("1
_output_shapes
:����������� "
pads

    "
group
�
layer1-maxpoolPool
layer0-act"
kernel_shape
"
pooling_typeMAX"
strides
"1
_output_shapes
:����������� "
pads

      
R

layer2-actRelulayer2-conv"1
_output_shapes
:�����������@
S
layer10-actRelulayer10-conv"0
_output_shapes
:���������88�
S
layer15-actRelulayer15-conv"0
_output_shapes
:����������
�
layer22-convConvlayer21-act"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:����������"
pads

    "
group
�
layer10-convConv
layer9-act"0
_output_shapes
:���������88�"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias(
�
layer7-maxpoolPool
layer6-act"
kernel_shape
"
pooling_typeMAX"
strides
"0
_output_shapes
:���������88�"
pads

      
�
layer14-convConvlayer13-act"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:����������"
pads

    "
group
S
layer19-actRelulayer19-conv"0
_output_shapes
:����������
Q

layer4-actRelulayer4-conv"0
_output_shapes
:���������pp�
R

layer0-actRelulayer0-conv"1
_output_shapes
:����������� 
�
layer27-reorg	DataInputlayer26-act"0
_output_shapes
:����������"%
shape:����������
�
layer3-maxpoolPool
layer2-act"
kernel_shape
"
pooling_typeMAX"
strides
"/
_output_shapes
:���������pp@"
pads

      
l
data	DataInput"1
_output_shapes
:�����������"&
shape:�����������
S
layer29-actRelulayer29-conv"0
_output_shapes
:����������
R
layer26-actRelulayer26-conv"/
_output_shapes
:���������@
�
layer18-convConvlayer17-maxpool"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias(
S
layer20-actRelulayer20-conv"0
_output_shapes
:����������
S
layer24-actRelulayer24-conv"0
_output_shapes
:����������
�
layer30-convConvlayer29-act"/
_output_shapes
:���������}"
pads

        "
group"
kernel_shape	
�}"
strides
"
use_bias(
�
layer29-convConvlayer28-concat"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

�
�"
strides
"
use_bias(
Q

layer9-actRelulayer9-conv"0
_output_shapes
:���������88�
�
layer19-convConvlayer18-act"0
_output_shapes
:����������"
pads

        "
group"
kernel_shape

��"
strides
"
use_bias(
�
layer12-convConvlayer11-maxpool"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias(
S
layer14-actRelulayer14-conv"0
_output_shapes
:����������
�
layer6-convConv
layer5-act"
kernel_shape	
@�"
strides
"
use_bias("0
_output_shapes
:���������pp�"
pads

    "
group
�
layer9-convConv
layer8-act"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:���������88�"
pads

        "
group
�
layer17-maxpoolPoollayer16-act"
kernel_shape
"
pooling_typeMAX"
strides
"0
_output_shapes
:����������"
pads

      
�
layer16-convConvlayer15-act"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:����������
�
layer15-convConvlayer14-act"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:����������"
pads

        "
group
�
layer24-convConvlayer23-act"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias(
�
layer11-maxpoolPoollayer10-act"
strides
"0
_output_shapes
:����������"
pads

      "
kernel_shape
"
pooling_typeMAX
S
layer12-actRelulayer12-conv"0
_output_shapes
:����������
�
layer13-convConvlayer12-act"0
_output_shapes
:����������"
pads

        "
group"
kernel_shape

��"
strides
"
use_bias(
�
layer20-convConvlayer19-act"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape

��"
strides
"
use_bias(
S
layer13-actRelulayer13-conv"0
_output_shapes
:����������
Q

layer8-actRelulayer8-conv"0
_output_shapes
:���������88�
S
layer22-actRelulayer22-conv"0
_output_shapes
:����������
r
layer28-concatConcatlayer27-reorglayer24-act"

axis"0
_output_shapes
:����������

S
layer21-actRelulayer21-conv"0
_output_shapes
:����������
S
layer16-actRelulayer16-conv"0
_output_shapes
:����������
�
layer4-convConvlayer3-maxpool"0
_output_shapes
:���������pp�"
pads

    "
group"
kernel_shape	
@�"
strides
"
use_bias(
�
layer2-convConvlayer1-maxpool"1
_output_shapes
:�����������@"
pads

    "
group"
kernel_shape
 @"
strides
"
use_bias(
P

layer5-actRelulayer5-conv"/
_output_shapes
:���������pp@
�
layer8-convConvlayer7-maxpool"
group"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:���������88�"
pads

    
�
layer26-convConvlayer16-act"/
_output_shapes
:���������@"
pads

        "
group"
kernel_shape	
�@"
strides
"
use_bias(
�
layer21-convConvlayer20-act"
kernel_shape

��"
strides
"
use_bias("0
_output_shapes
:����������"
pads

        "
group
S
layer18-actRelulayer18-conv"0
_output_shapes
:����������