
╝
conv4_4/dwiseConvrelu4_4/expand"0
_output_shapes
:         -<ђ"
pads

    "
groupђ"
kernel_shape

ђђ"
strides
"
use_bias(
Z
fc7_mbox_loc_flatFlattenfc7_mbox_loc_perm")
_output_shapes
:         ░Ђ
╝
conv5_2/dwiseConvrelu5_2/expand"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

    "
group└
Ћ
conv4_3_norm_mbox_loc_perm	DataInputconv4_3_norm_mbox_loc"/
_output_shapes
:         < -"$
shape:         < -
┤
conv6_2_mbox_confConvrelu7"
strides
"
use_bias("/
_output_shapes
:         B"
pads

    "
group"
kernel_shape	
ђ
B
Ї
conv8_2_mbox_conf_perm	DataInputconv8_2_mbox_conf"/
_output_shapes
:         ,"$
shape:         ,
╝
conv3_2/dwiseConvrelu3_2/expand"0
_output_shapes
:         -<љ"
pads

    "
groupљ"
kernel_shape

љљ"
strides
"
use_bias(
╗
conv4_4/expandConvconv4_3/linear"
kernel_shape	
@ђ"
strides
"
use_bias("0
_output_shapes
:         -<ђ"
pads

        "
group
V
relu5_2/dwiseReluconv5_2/dwise"0
_output_shapes
:         └
и
conv6_3/expandConv	block_6_2"0
_output_shapes
:         └"
pads

        "
group"
kernel_shape

а└"
strides
"
use_bias(
▓
conv2_1/expandConvrelu1"
kernel_shape
  "
strides
"
use_bias("1
_output_shapes
:         ┤­ "
pads

        "
group
Х
conv4_6/expandConv	block_4_5"0
_output_shapes
:         -<ђ"
pads

        "
group"
kernel_shape	
@ђ"
strides
"
use_bias(
X
relu6_3/expandReluconv6_3/expand"0
_output_shapes
:         └
a
	block_4_1Addconv3_2/linearconv4_1/linear"/
_output_shapes
:         -< 
X
relu4_6/expandReluconv4_6/expand"0
_output_shapes
:         -<ђ
╗
conv7_2_mbox_confConvconv6_2_relu"/
_output_shapes
:         B"
pads

    "
group"
kernel_shape	
ђB"
strides
"
use_bias(
Х
conv4_7/expandConv	block_4_6"
strides
"
use_bias("0
_output_shapes
:         -<ђ"
pads

        "
group"
kernel_shape	
@ђ
a
	block_4_4Addconv4_3/linearconv4_4/linear"/
_output_shapes
:         -<@
│
conv7_2Convconv7_1_relu"
strides
"
use_bias("0
_output_shapes
:         ђ"
pads

    "
group"
kernel_shape

ђђ
X
relu4_3/expandReluconv4_3/expand"0
_output_shapes
:         -<└
Ј
conv7_2_mbox_priorbox	DataInputconv6_2_reludata"0
_output_shapes
:         ђ	"%
shape:         ђ	
\
	block_5_2Add	block_5_1conv5_2/linear"/
_output_shapes
:         `
Ї
conv6_2_mbox_conf_perm	DataInputconv6_2_mbox_conf"/
_output_shapes
:         B"$
shape:         B
X
relu3_1/expandReluconv3_1/expand"0
_output_shapes
:         Zxљ
И
conv2_2/dwiseConvrelu2_2/expand"/
_output_shapes
:         Zx`"
pads

    "
group`"
kernel_shape
``"
strides
"
use_bias(
U
relu2_2/dwiseReluconv2_2/dwise"/
_output_shapes
:         Zx`
Ј
fc7_mbox_priorbox	DataInputrelu5_3/expanddata"1
_output_shapes
:         ░Ђ"&
shape:         ░Ђ
╝
conv4_1/dwiseConvrelu4_1/expand"
group└"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         -<└"
pads

    
╝
conv5_1/dwiseConvrelu5_1/expand"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

    "
group└
╗
conv8_2_mbox_confConvconv7_2_relu"
strides
"
use_bias("/
_output_shapes
:         ,"
pads

    "
group"
kernel_shape	
ђ,
╣
conv4_3/linearConvrelu4_3/dwise"/
_output_shapes
:         -<@"
pads

        "
group"
kernel_shape	
└@"
strides
"
use_bias(
V
relu3_1/dwiseReluconv3_1/dwise"0
_output_shapes
:         Zxљ
V
relu4_5/dwiseReluconv4_5/dwise"0
_output_shapes
:         -<ђ
Y
relu2_1/expandReluconv2_1/expand"1
_output_shapes
:         ┤­ 
И
conv2_2/linearConvrelu2_2/dwise"/
_output_shapes
:         Zx"
pads

        "
group"
kernel_shape
`"
strides
"
use_bias(
┼
mbox_locConcatconv4_3_norm_mbox_loc_flatfc7_mbox_loc_flatconv6_2_mbox_loc_flatconv7_2_mbox_loc_flatconv8_2_mbox_loc_flat"1
_output_shapes
:         лл"

axis
G
relu1Reluconv1"1
_output_shapes
:         ┤­ 
V
relu4_4/dwiseReluconv4_4/dwise"0
_output_shapes
:         -<ђ
a
conv7_2_mbox_loc_flatFlattenconv7_2_mbox_loc_perm"(
_output_shapes
:         ђ	
ѕ
conv6_2_mbox_priorbox	DataInputrelu7data"0
_output_shapes
:         Я!"%
shape:         Я!
V
relu6_2/dwiseReluconv6_2/dwise"0
_output_shapes
:         └
b
	block_6_1Addconv5_3/linearconv6_1/linear"0
_output_shapes
:         а
И
fc7_mbox_locConvrelu5_3/expand"/
_output_shapes
:         "
pads

    "
group"
kernel_shape	
└"
strides
"
use_bias(
│
conv6_2Convconv6_1_relu"
kernel_shape

ђђ"
strides
"
use_bias("0
_output_shapes
:         ђ"
pads

    "
group
и
conv6_2/expandConv	block_6_1"
kernel_shape

а└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

        "
group
Ќ
conv4_3_norm_mbox_conf_perm	DataInputconv4_3_norm_mbox_conf"/
_output_shapes
:         <X-"$
shape:         <X-
║
conv2_1/linearConvrelu2_1/dwise"1
_output_shapes
:         ┤­"
pads

        "
group"
kernel_shape
 "
strides
"
use_bias(
╝
conv4_2/dwiseConvrelu4_2/expand"
group└"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         -<└"
pads

    
X
relu6_2/expandReluconv6_2/expand"0
_output_shapes
:         └
l
data	DataInput"1
_output_shapes
:         УЯ"&
shape:         УЯ
╗
conv5_1/expandConvconv4_7/linear"0
_output_shapes
:         └"
pads

        "
group"
kernel_shape	
`└"
strides
"
use_bias(
Ј
conv8_2_mbox_priorbox	DataInputconv7_2_reludata"0
_output_shapes
:         └"%
shape:         └
ў
conv4_3_norm_mbox_priorbox	DataInputrelu4_7/expanddata"1
_output_shapes
:         ђБ"&
shape:         ђБ
c
conv6_2_mbox_conf_flatFlattenconv6_2_mbox_conf_perm"(
_output_shapes
:         У\
╗
conv6_1/linearConvrelu6_1/dwise"0
_output_shapes
:         а"
pads

        "
group"
kernel_shape

└а"
strides
"
use_bias(
X
relu5_2/expandReluconv5_2/expand"0
_output_shapes
:         └
Х
conv4_5/expandConv	block_4_4"0
_output_shapes
:         -<ђ"
pads

        "
group"
kernel_shape	
@ђ"
strides
"
use_bias(
V
relu4_7/dwiseReluconv4_7/dwise"0
_output_shapes
:         ђ
╣
conv3_1/linearConvrelu3_1/dwise"
kernel_shape	
љ"
strides
"
use_bias("/
_output_shapes
:         Zx"
pads

        "
group
X
relu4_2/expandReluconv4_2/expand"0
_output_shapes
:         -<└
╝
conv6_1/dwiseConvrelu6_1/expand"0
_output_shapes
:         └"
pads

    "
group└"
kernel_shape

└└"
strides
"
use_bias(
O
conv6_2_reluReluconv6_2"0
_output_shapes
:         ђ
Ё
fc7_mbox_conf_perm	DataInputfc7_mbox_conf"/
_output_shapes
:         B"$
shape:         B
V
relu6_3/dwiseReluconv6_3/dwise"0
_output_shapes
:         └
a
	block_5_1Addconv4_7/linearconv5_1/linear"/
_output_shapes
:         `
a
conv6_2_mbox_loc_flatFlattenconv6_2_mbox_loc_perm"(
_output_shapes
:         Я!
Х
conv3_2/expandConv	block_3_1"0
_output_shapes
:         Zxљ"
pads

        "
group"
kernel_shape	
љ"
strides
"
use_bias(
V
relu5_3/dwiseReluconv5_3/dwise"0
_output_shapes
:         └
V
relu3_2/dwiseReluconv3_2/dwise"0
_output_shapes
:         -<љ
І
conv7_2_mbox_loc_perm	DataInputconv7_2_mbox_loc"/
_output_shapes
:         "$
shape:         
O
conv6_1_reluReluconv6_1"0
_output_shapes
:         ђ
W
relu2_1/dwiseReluconv2_1/dwise"1
_output_shapes
:         ┤­ 
╣
conv4_6/linearConvrelu4_6/dwise"
strides
"
use_bias("/
_output_shapes
:         -<@"
pads

        "
group"
kernel_shape	
ђ@
╝
conv4_6/dwiseConvrelu4_6/expand"
kernel_shape

ђђ"
strides
"
use_bias("0
_output_shapes
:         -<ђ"
pads

    "
groupђ
a
	block_3_1Addconv2_2/linearconv3_1/linear"/
_output_shapes
:         Zx
╣
conv4_4/linearConvrelu4_4/dwise"
kernel_shape	
ђ@"
strides
"
use_bias("/
_output_shapes
:         -<@"
pads

        "
group
Ї
conv7_2_mbox_conf_perm	DataInputconv7_2_mbox_conf"/
_output_shapes
:         B"$
shape:         B
╝
conv3_1/dwiseConvrelu3_1/expand"
kernel_shape

љљ"
strides
"
use_bias("0
_output_shapes
:         Zxљ"
pads

    "
groupљ
H
relu7Reluconv6_4"0
_output_shapes
:         ђ

X
relu4_4/expandReluconv4_4/expand"0
_output_shapes
:         -<ђ
Х
conv4_2/expandConv	block_4_1"0
_output_shapes
:         -<└"
pads

        "
group"
kernel_shape	
 └"
strides
"
use_bias(
Х
conv5_3/expandConv	block_5_2"
strides
"
use_bias("0
_output_shapes
:         └"
pads

        "
group"
kernel_shape	
`└
V
relu4_1/dwiseReluconv4_1/dwise"0
_output_shapes
:         -<└
╣
conv4_2/linearConvrelu4_2/dwise"
strides
"
use_bias("/
_output_shapes
:         -< "
pads

        "
group"
kernel_shape	
└ 
І
conv8_2_mbox_loc_perm	DataInputconv8_2_mbox_loc"/
_output_shapes
:         "$
shape:         
l
conv4_3_norm_mbox_loc_flatFlattenconv4_3_norm_mbox_loc_perm")
_output_shapes
:         ђБ
І
conv6_2_mbox_loc_perm	DataInputconv6_2_mbox_loc"/
_output_shapes
:         "$
shape:         
╣
conv4_7/linearConvrelu4_7/dwise"/
_output_shapes
:         `"
pads

        "
group"
kernel_shape	
ђ`"
strides
"
use_bias(
X
relu5_1/expandReluconv5_1/expand"0
_output_shapes
:         └
╝
conv4_5/dwiseConvrelu4_5/expand"0
_output_shapes
:         -<ђ"
pads

    "
groupђ"
kernel_shape

ђђ"
strides
"
use_bias(
х
conv6_4Convconv6_3/linear"0
_output_shapes
:         ђ
"
pads

        "
group"
kernel_shape

└ђ
"
strides
"
use_bias(
n
conv4_3_norm_mbox_conf_flatFlattenconv4_3_norm_mbox_conf_perm")
_output_shapes
:         а└
╗
conv4_1/expandConvconv3_2/linear"
group"
kernel_shape	
 └"
strides
"
use_bias("0
_output_shapes
:         -<└"
pads

        
]
	block_6_2Add	block_6_1conv6_2/linear"0
_output_shapes
:         а
Х
conv5_2/expandConv	block_5_1"
kernel_shape	
`└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

        "
group
╗
conv5_3/linearConvrelu5_3/dwise"
strides
"
use_bias("0
_output_shapes
:         а"
pads

        "
group"
kernel_shape

└а
X
relu3_2/expandReluconv3_2/expand"0
_output_shapes
:         Zxљ
V
relu4_2/dwiseReluconv4_2/dwise"0
_output_shapes
:         -<└
c
conv8_2_mbox_conf_flatFlattenconv8_2_mbox_conf_perm"(
_output_shapes
:         љ
┬
conv4_3_norm_mbox_confConvrelu4_7/expand"
kernel_shape	
ђX"
strides
"
use_bias("/
_output_shapes
:         -<X"
pads

    "
group
│
conv6_2_mbox_locConvrelu7"/
_output_shapes
:         "
pads

    "
group"
kernel_shape	
ђ
"
strides
"
use_bias(
╗
conv3_1/expandConvconv2_2/linear"
group"
kernel_shape	
љ"
strides
"
use_bias("0
_output_shapes
:         Zxљ"
pads

        
┴
conv4_3_norm_mbox_locConvrelu4_7/expand"
kernel_shape	
ђ "
strides
"
use_bias("/
_output_shapes
:         -< "
pads

    "
group
Х
conv4_3/expandConv	block_4_2"
strides
"
use_bias("0
_output_shapes
:         -<└"
pads

        "
group"
kernel_shape	
 └
╣
conv4_1/linearConvrelu4_1/dwise"/
_output_shapes
:         -< "
pads

        "
group"
kernel_shape	
└ "
strides
"
use_bias(
Y
relu2_2/expandReluconv2_2/expand"1
_output_shapes
:         ┤­`
\
	block_4_6Add	block_4_5conv4_6/linear"/
_output_shapes
:         -<@
\
	block_4_5Add	block_4_4conv4_5/linear"/
_output_shapes
:         -<@
Ѓ
fc7_mbox_loc_perm	DataInputfc7_mbox_loc"/
_output_shapes
:         "$
shape:         
╗
conv2_2/expandConvconv2_1/linear"
group"
kernel_shape
`"
strides
"
use_bias("1
_output_shapes
:         ┤­`"
pads

        
╦
	mbox_confConcatconv4_3_norm_mbox_conf_flatfc7_mbox_conf_flatconv6_2_mbox_conf_flatconv7_2_mbox_conf_flatconv8_2_mbox_conf_flat"1
_output_shapes
:         ▄Ю"

axis
е
conv1Convdata"
kernel_shape
 "
strides
"
use_bias("1
_output_shapes
:         ┤­ "
pads

    "
group
│
conv7_1Convconv6_2_relu"0
_output_shapes
:         ђ"
pads

        "
group"
kernel_shape

ђђ"
strides
"
use_bias(
╗
conv6_2/linearConvrelu6_2/dwise"
strides
"
use_bias("0
_output_shapes
:         а"
pads

        "
group"
kernel_shape

└а
╣
conv3_2/linearConvrelu3_2/dwise"
kernel_shape	
љ "
strides
"
use_bias("/
_output_shapes
:         -< "
pads

        "
group
O
conv7_1_reluReluconv7_1"0
_output_shapes
:         ђ
╝
conv5_3/dwiseConvrelu5_3/expand"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

    "
group└
V
relu5_1/dwiseReluconv5_1/dwise"0
_output_shapes
:         └
O
conv7_2_reluReluconv7_2"0
_output_shapes
:         ђ
X
relu5_3/expandReluconv5_3/expand"0
_output_shapes
:         └
c
conv7_2_mbox_conf_flatFlattenconv7_2_mbox_conf_perm"(
_output_shapes
:         Я
V
relu4_6/dwiseReluconv4_6/dwise"0
_output_shapes
:         -<ђ
X
relu4_5/expandReluconv4_5/expand"0
_output_shapes
:         -<ђ
║
conv2_1/dwiseConvrelu2_1/expand"1
_output_shapes
:         ┤­ "
pads

    "
group "
kernel_shape
  "
strides
"
use_bias(
╝
conv6_1/expandConvconv5_3/linear"
strides
"
use_bias("0
_output_shapes
:         └"
pads

        "
group"
kernel_shape

а└
X
relu4_1/expandReluconv4_1/expand"0
_output_shapes
:         -<└
╝
conv4_7/dwiseConvrelu4_7/expand"0
_output_shapes
:         ђ"
pads

    "
groupђ"
kernel_shape

ђђ"
strides
"
use_bias(
\
	block_4_2Add	block_4_1conv4_2/linear"/
_output_shapes
:         -< 
╝
conv4_3/dwiseConvrelu4_3/expand"
strides
"
use_bias("0
_output_shapes
:         -<└"
pads

    "
group└"
kernel_shape

└└
╣
fc7_mbox_confConvrelu5_3/expand"/
_output_shapes
:         B"
pads

    "
group"
kernel_shape	
└B"
strides
"
use_bias(
╗
conv6_3/linearConvrelu6_3/dwise"
strides
"
use_bias("0
_output_shapes
:         └"
pads

        "
group"
kernel_shape

└└
V
relu4_3/dwiseReluconv4_3/dwise"0
_output_shapes
:         -<└
V
relu6_1/dwiseReluconv6_1/dwise"0
_output_shapes
:         └
г
conv6_1Convrelu7"
kernel_shape

ђ
ђ"
strides
"
use_bias("0
_output_shapes
:         ђ"
pads

        "
group
\
fc7_mbox_conf_flatFlattenfc7_mbox_conf_perm")
_output_shapes
:         Сс
╣
conv5_2/linearConvrelu5_2/dwise"/
_output_shapes
:         `"
pads

        "
group"
kernel_shape	
└`"
strides
"
use_bias(
a
conv8_2_mbox_loc_flatFlattenconv8_2_mbox_loc_perm"(
_output_shapes
:         └
X
relu4_7/expandReluconv4_7/expand"0
_output_shapes
:         -<ђ
╣
conv4_5/linearConvrelu4_5/dwise"
strides
"
use_bias("/
_output_shapes
:         -<@"
pads

        "
group"
kernel_shape	
ђ@
║
conv8_2_mbox_locConvconv7_2_relu"
strides
"
use_bias("/
_output_shapes
:         "
pads

    "
group"
kernel_shape	
ђ
║
conv7_2_mbox_locConvconv6_2_relu"
strides
"
use_bias("/
_output_shapes
:         "
pads

    "
group"
kernel_shape	
ђ
X
relu6_1/expandReluconv6_1/expand"0
_output_shapes
:         └
╩
mbox_priorboxConcatconv4_3_norm_mbox_priorboxfc7_mbox_priorboxconv6_2_mbox_priorboxconv7_2_mbox_priorboxconv8_2_mbox_priorbox"1
_output_shapes
:         лл"

axis
╝
conv6_3/dwiseConvrelu6_3/expand"
strides
"
use_bias("0
_output_shapes
:         └"
pads

    "
group└"
kernel_shape

└└
╝
conv6_2/dwiseConvrelu6_2/expand"
kernel_shape

└└"
strides
"
use_bias("0
_output_shapes
:         └"
pads

    "
group└
╣
conv5_1/linearConvrelu5_1/dwise"/
_output_shapes
:         `"
pads

        "
group"
kernel_shape	
└`"
strides
"
use_bias(