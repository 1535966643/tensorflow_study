
import tensorflow as tf

tf.reset_default_graph()


logdir = 'D:/log'

#  add_n 表示两个列表的数据相加 random_uniform[a, b]:产生满足均匀分布的随机矩阵a行b列
intput1 = tf.constant([1.0,2.0,3.0], name='intput1')
intput2 = tf.Variable(tf.random_uniform([3]), name='intput2')
output = tf.add_n([intput1, intput2], name='add')

# with tf.Session() as sess:
	# print(sess.run(tf.random_uniform([3, 2])))

writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()


# tensorboard --logdir=D:\log (进入目录后输入这条语句) 