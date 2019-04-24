

import tensorflow as tf
import numpy as np

'''
	方法定义
	tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
	**input : ** 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
	filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
	strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
	padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
	use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
'''


# [batch, in_heiht,in_width,in_channels] 分别表示：样本的个数，图像的长和宽，图像的通道数
input_data = tf.Variable(np.random.rand(10,9,9,4), dtype=np.float32)	# []and() 是使用tf和numpy之间的产生随机正太数的不同写法
# [in_heiht,in_width,in_channels，out_channels]分别表示：图像的尺寸，输入的通道，输出的通道
filter_data = tf.Variable(np.random.rand(3,3,4,2), dtype=np.float32)
# padding：SAME表示边缘填充， VALID表示边缘不填充
# y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='SAME')
y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='VALID')

print(input_data)
print(y)

# ***重点*** 通道数：在输入层图像为灰色或彩色，通道数为1或3，但是在后面通道就是指一张图，
# 	若：[in_channel,out_channel]=(3, 32) 输入为彩色图，卷积输出后的那层有32个特征
#         (由32个卷积核提取出来)，即输出后的图深度为32

'''
	池化部分
'''
input_data = tf.Variable(np.random.rand(10,6,6,4), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2,2,4,2), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='SAME')

# 最大池化
# ksize=[1,2,2,1] 表示窗口大小为2x2
output = tf.nn.max_pool(value=y, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
# 平均池化
# output = tf.nn.avg_pool(value=y, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

print('conv:', y)
print('pooling:', output)


