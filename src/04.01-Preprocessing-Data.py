# %%
# 内容：如何运用sklearn 包中的preprocessing进行数据处理
# 1 特征标准化，数据缩到拥有0均值和单位方差
# 2 特征归一化，使样本拥有单位范数
# 3 将数据缩放当一定范围
# 4 特征二值化
# 5 缺失数据处理
from sklearn import impute
from numpy import nan
from sklearn import preprocessing
import numpy as np

# 初始值，一个3×3的矩阵
X = np.array([[1., -2., 2.],
              [3., 0., 0.],
              [0., 1., -1.]])

# 特征标准化，拥有零均值和单位方差
X_scaled = preprocessing.scale(X)
print(X_scaled)

# %%
# 检验其均值
X_scaled.mean(axis=0)

# %%
# 检验其标准差
X_scaled.std(axis=0)

# %%
# 特征归一化，缩放单个样本，使其具有单位范数的过程
# norm='l1'，曼哈顿距离，绝对值相加为1
X_normalized_l1 = preprocessing.normalize(X, norm='l1')
X_normalized_l1

# %%
# norm='l2'，欧式距离，平方和相加为1
X_normalized_l2 = preprocessing.normalize(X, norm='l2')
X_normalized_l2

# %%
# 将特征缩放到一定范围，默认为(0,1)，（特征最大的绝对值）
min_max_scaler = preprocessing.MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(X)
X_min_max
# %%
# 也可以指定范围
min_max_scaler2 = preprocessing.MinMaxScaler(feature_range=(-10, 10))
X_min_max2 = min_max_scaler2.fit_transform(X)
X_min_max2

# %%
# 4. 特征二值化
binarizer = preprocessing.Binarizer(threshold=0.5)
X_binarized = binarizer.transform(X)
print(X_binarized)

# %%
# 5 数据缺失处理
X = np.array([[nan, 0, 3], [2, 9, -8], [1, nan, 1], [5, 2, 4], [7, 6, -3]])
imp = impute.SimpleImputer(strategy='mean')
# imp = impute.SimpleImputer(strategy='median')
X2 = imp.fit_transform(X)
print(X2)