# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 08:23:20 2019

@author: huoqs
"""

#%%
# 导入numpy，sklearn，matplotlib等必要的包
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#%%
# 导入波士顿数据集，里面包含了波士顿房价信息
boston = datasets.load_boston()

#%%
# 显示boston数据集的属性
dir(boston)

#%%
boston.data.shape
#%%
boston.target.shape

#%%
# 创建线性回归模型
# linreg = linear_model.LinearRegression()
linreg = linear_model.Lasso()
# linreg = linear_model.RidgeRegression()

#%%
# 拆分训练集和测试集
X_train, X_test, y_train, y_test = modsel.train_test_split(boston.data, boston.target, 
                                                           test_size = 0.1, random_state = 42)

#%%
# 开始训练，用fit函数
linreg.fit(X_train, y_train)

#%%
# 评估平均方差
metrics.mean_squared_error(y_train, linreg.predict(X_train))

#%%
linreg.score(X_train, y_train)

#%%
# 测试集进行验证
y_pred = linreg.predict(X_test)
print(y_pred)
metrics.mean_squared_error(y_test, y_pred)

#%%
# 绘制测试集数据
plt.figure(figsize = (10, 6))
plt.plot(y_test, linewidth=3, label="ground truth")
plt.plot(y_pred, linewidth=3, label="predicted")
plt.legend(loc="best")
plt.xlabel('test data points')
plt.ylabel('target value')

#%%
# 这些函数还不是特别明白用法
plt.plot(y_test, y_pred, 'o')
plt.plot([-10, 60],[-10, 60], 'k--')
plt.axis([-10, 60,-10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')
scorestr = r'R$^2$ = %.3f' % linreg.score(X_test, y_test)
errstr = 'MSE = %3.f' % metrics.mean_squared_error(y_test, y_pred)
plt.text(-5, 50, scorestr, fontsize = 12) 
plt.text(-5, 45, errstr, fontsize = 12)


# %%
