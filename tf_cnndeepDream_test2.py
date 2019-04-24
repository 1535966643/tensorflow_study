
# 最终的一点效果
from __future__ import print_function
import os
from io import BytesIO
import numpy  as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf

# 调整图像的尺寸
def resize(img, hw):
	min = img.min()
	max = img.max()
	img = (img - min) / (max - min) * 255
	img = np.float32(scipy.misc.imresize(img, hw))
	img = img / 255 * (max - min) + min
	return img

# 原始图像的尺寸可能很大，从而导致内存耗尽问题
# 图像大小进行计算梯度，避免内存问题
def calc_grad_tiled(img, t_grad, tile_size=512):
	sz = tile_size
	h,w = img.shape[:2]
	sx, sy = np.random.randint(sz, size=2)
	# 先在行上做移动，再在列上做移动
	img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
	grad = np.zeros_like(img)
	for y in range(0,max(h - sz // 2, sz), sz):
		for x in range(0, max(w - sz // 2, sz), sz):
			sub = img_shift[y:y + sz, x:x+sz]
			g = sess.run(t_grad,{t_input:sub})
			grad[y:y+sz, x:x+sz] = g
	return np.roll(np.roll(grad, -sx, 1), -sy, 0)

# 改进之后的渲染函数
# def render_deepdream(t_obj, step=1.0, octave_n=4, octave_scale=1.4):
def render_deepdream(t_obj, img0, number, step=1.5, octave_n=4, octave_scale=1.4):
	t_score = tf.reduce_mean(t_obj)   # 求平均值
	# 计算t_score对t_input的梯度
	t_grad = tf.gradients(t_score, t_input)[0]
	img = img0.copy()

	octaves = []
	for i in range(octave_n - 1):
		hw = img.shape[:2]
		lo = resize(img, np.int32(np.float32(hw)/octave_scale))
		hi = img - resize(lo, hw)
		img = lo
		octaves.append(hi)

	# 首先生成低频的图像，再一次放大并且加上高频
	for octave in range(octave_n):
		if octave > 0:
			hi = octaves[-octave]
			img = resize(img, hi.shape[:2]) + hi
			for i in range(10):
				g = calc_grad_tiled(img, t_grad)
				img += g * (step/(np.abs(g).mean() + 1e-7))
	
	img = img.clip(0,255)
	savePath = 'c:/testimage/deepdream{}.jpg'.format(number)
	scipy.misc.toimage(img).save(savePath)
	# im = PIL.Image.open('c:/mountain_deepDream.jpg').show()


if __name__ == '__main__':
	
	# 创建图和会话
	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	model_fn = 'D:/tensorflow_inception_graphData/tensorflow_inception_graph.pb'
	with tf.gfile.FastGFile(model_fn,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# 定义输入图像的占位符
	t_input = tf.placeholder(np.float32, name='input')
	# 图像预处理--减均值
	imagenet_mean = 117.0
	# 头像那个预处理--增加维度
	t_preproessed = tf.expand_dims(t_input - imagenet_mean, 0)
	# 导入模型并且将与处理的图像送入网络
	tf.import_graph_def(graph_def,{'input':t_preproessed})

	name='mixed4c'
	layer_output = graph.get_tensor_by_name('import/%s:0'%name)
	for x in range(1,9):
		img0 = PIL.Image.open('c:/testimage/{}.jpg'.format(x))
		img0 = np.float32(img0)
		render_deepdream(tf.square(layer_output), img0, x)









