
import tensorflow as tf
import numpy as np

x = np.array([-3, 1, 1, 8, 9, 7, -2.5])
# 分类的概率
pred  = tf.nn.softmax(x)

sess = tf.Session()
v = sess.run(pred)
print(v)

# 多类别
# 交叉熵的使用
'''
H = -E p(x)log q(x)  （p、q 都是指的是概率值）
表示 用q去预测p
'''


sess.close()


