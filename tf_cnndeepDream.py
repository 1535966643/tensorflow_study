'''
	tensorflow 中简单图像的处理
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile('c:/1.jpg','rb').read()
# with 执行是按照“小块”来进行的
with tf.Session() as sess:
	# 对图像解码从而得到图像对应的三位矩阵
	img_data = tf.image.decode_jpeg(image_raw_data)
	# 张量  所以使用.eval
	# print(img_data.eval())

	# plt.imshow(img_data.eval())
	# plt.show()

# 图像的缩放
with tf.Session() as sess:
	# 双线性插值法缩放图像
	img_data = tf.image.decode_jpeg(image_raw_data)
	img1 = tf.image.resize_images(img_data, [256,256], method=0)
	# 用最近邻插值法缩放图像
	# img1 = tf.image.resize_images(img_data, [256,256], method=1)
	# # 双立方插值法
	# img1 = tf.image.resize_images(img_data, [256,256], method=2)
	# # 区域插值法
	# img1 = tf.image.resize_images(img_data, [256,256], method=3)

	img = np.asarray(img1.eval(), dtype='uint8')
	plt.imshow(img)
	plt.show()

# 图像进行裁剪(中央地方)
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	crop = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
	plt.imshow(crop.eval())
	plt.show()
# 还有随机裁剪
	
# 图像水平翻转
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	plt.imshow(img_data.eval())
	# plt.axis('off')

	flip_left_right = tf.image.flip_left_right(img_data)
	plt.imshow(flip_left_right.eval())
	plt.axis('off')
	plt.show()


# 图像上下翻转
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	flip_up_down = tf.image.flip_up_down(img_data)
	plt.imshow(flip_up_down.eval())
	plt.show()

# 改边图像的对比度
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	# 将图像的对比度降低至到原来的1/2
	# contrast = tf.image.adjust_contrast(img_data, 0.5)
	# 将图像的对比度提高至到原来的5
	# contrast = tf.image.adjust_contrast(img_data, 5)
	# # 在[lower, uppre]范围随机调整对比度
	contrast = tf.image.random_contrast(img_data, lower=0.2, upper=3)

	plt.imshow(contrast.eval())
	plt.show()

# 白化处理
# 将图像的像素值转化为零均值和单位方差
# 函数： tf.image.per_image_standardization(img_data)



# 模型社区 TensorFlow model zoo



# 使用了dropout
