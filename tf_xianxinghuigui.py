import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


np.random.seed(5)
# 等差数列生成100个数据
x_data = np.linspace(-1,1,100)		
print(*x_data.shape)
# 产生x_data.shape个满足标准正太分布的随机数
y_data = x_data * 2.0 + 1.0 + np.random.randn(*x_data.shape) * 0.4
# 产生散点
plt.scatter(x_data, y_data)
plt.plot(x_data, x_data * 2 +1, color='red')






plt.show()


