import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
x = np.linspace(-5,5,20)
y = 0.5*x + 3
for i in range(0,20):
    noise = random.uniform(-0.5,0.5)  #生成随机数
    y[i]+=noise                       #修改y的值 （随机扰动）

reg = LinearRegression()
reg.fit(x.reshape(-1,1),y)
plt.scatter(x,y,color='orange')
plt.plot(x,reg.predict(x.reshape(-1,1)))
plt.show()
