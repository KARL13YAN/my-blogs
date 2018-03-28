# Python机器学习读书笔记（四）特征抽取

说明：

1. 关于本书: [《Python机器学习》](https://book.douban.com/subject/27000110/)
2. 本笔记侧重代码调用，只描述了一些简单概念，本书的公式推导不在这里展示
3. 本页[代码](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/ch05.ipynb)

---

**特征抽取** 可以将原始数据集变换到一个维度更低的新的特征子空间，在尽可能多地保持相关信息的情况下，对数据进行压缩。

## 1. 主成份分析 Principle Component Analysis, PCA

### 1.1 **一些博客**

- [博客1](http://www.cnblogs.com/pinard/p/6239403.html)
- [博客2](http://blog.csdn.net/zhongkelee/article/details/44064401)
- [博客3](http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html)
- [一个例子](http://www.cnblogs.com/zhangchaoyang/articles/2222048.html)
- [参考1](https://www.jiqizhixin.com/articles/2017-07-05-2)
- [参考2](https://www.jiqizhixin.com/articles/2017-08-31-2)

### 1.2 **简单介绍**

- **无监督算法**

- 如果我们将矩阵看作物理运动，那么最重要的就是运动方向（特征向量）和速度（特征值）。因为物理运动只需要方向和速度就可以描述，同理矩阵也可以仅使用特征向量和特征值描述。

- PCA 是一种寻找高维数据（图像等）模式的工具。机器学习实践上经常使用 PCA 对输入神经网络的数据进行预处理。通过聚集、旋转和缩放数据，PCA 算法可以去除一些低方差的维度而达到降维的效果，这样操作能提升神经网络的收敛速度和整体效果。
- **目标** ：在高维数据中找到方差最大的方向，并将数据映射到一个维度不大于原始数据的新的子空间上。
- **如下图所示：** 以新的坐标是相互正交为约束条件，新的子空间上正交的坐标轴可以被解释为方差最大的方向。在这里$x_1, x_2$为原始特征的坐标轴，而$PC1, PC2$为主成份

![主成份示例图1](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_01.png)

### 1.3 **原理简介**

- 如果使用PCA降维，我们将构建一个$d*k$的**转换矩阵W**，这样就可以将一个样本向量x映射到新的k维子空间上去，且 $k<d$ ：

$$
x=[x_1, x_2, ..., x_d],  x\in R^d
$$

$$
xW, W\in R^{d*k}
$$

$$
z=[z_1, z_2, ..., z_d],  z\in R^k
$$

- 算法流程：

> 1. 对原始的d维数据做标准化处理
> 2. 构造样本协方差矩阵
> 3. 计算协方差矩阵的特征值和特征向量
> 4. 选择与前k个最大特征值对应的特征向量，其中k为新特征空间的维度
> 5. 通过前k个特征向量构建映射矩阵W
> 6. 通过映射矩阵W将d维的输入数据集X转换到新的k维特征子空间

- 协方差矩阵（$d*d$）是沿主对角线对称的，该矩阵成对的存储了不同特征之间的协方差。
- 两个特征之间的协方差若为**正，则同时增减**，反之两个特征之间的协方差若为**负，则朝相反方向变动**。
- 协方差矩阵的特征向量代表主成份（最大方差方向），而对应的特征值大小就决定了特征向量的重要性。

### 1.4 **示例**

- 加载数据集

```pyhton
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
```

- 划分数据集

```py
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
```

- 标准化

```py
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```

- 构造协方差矩阵，eigen_vals, eigen_vecs分别是协方差向量和协方差矩阵（d*d）

```py
import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
```

- 为了挑选前k个特征向量，绘制特征值的**方差贡献率**图像。特征值 $\lambda_j$ 的方差贡献率是指**特征值$\lambda_j$** 与 **所有特征值的和**的比值。使用Numpy的cumsum函数，计算累积方差：

```py
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


import matplotlib.pyplot as plt

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/pca1.png', dpi=300)
plt.show()
```

![效果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_02.png)

- 接着，按特征值的降序排列特征对,并选取两个对应特征值最大的特征向量

```py
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
```

- 将数据映射到子空间

```py
X_train_pca = X_train_std[0].dot(w)
```

- 可视化展示新数据：

```py
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca2.png', dpi=300)
plt.show()
```

![效果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_03.png)

### 1.5 **使用sklearn进行主成份分析**

使用了[前面用到的plot_decision_regions函数](http://blog.csdn.net/weixin_40604987/article/details/79566459)

```py
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca3.png', dpi=300)
plt.show()
```

![效果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_04.png)

- n_components=None时，可以保留所有主成份，并且可以通过explained_variance_ratio_属性获得相应的方差贡献率

```py
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

array([ 0.37329648,  0.18818926,  0.10896791,  0.07724389,  0.06478595,0.04592014,  0.03986936,  0.02521914,  0.02258181,  0.01830924,0.01635336,  0.01284271,  0.00642076])
```

## 2. 线性判别分析 Linear Discriminate Analysis, LDA

### 2.1 简单介绍

- LDA是一种可作为**特征抽取**的技术

- 可以提高数据分析过程中的计算效率
- 对于不适用与正则化的模型，可以降低因维度灾难带来的过拟合
- **监督算法**
- **目标**：发现可以最优化分类的特征子空间

![LDA解释图1](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_06.png)

- 如图所示，在x轴方向，通过线性判定，可以很好的将呈正态分布的两个类分开

- 虽然沿y轴方向的线性判定保持了数据集的较大方差，但是无法提供关于类别区分的任何信息，因此它不是一个好的线性判定

### 2.2 算法

- **思想**：给定训练集样例，设法将样例投影到一条直线上，使得同**类样例的投影尽可能接近**，**异类样例的投影点尽可能原理**；在对新的样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别。（下图截自 周志华《机器学习》）

![LDA](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/LDA.png )

- **假设**：

1. 数据呈正态分布
2. 各类别数据具有相同的协方差矩阵
3. 样本的特征从统计上来说相互独立
4. 事实上，即使违背上述假设，LDA仍能正常工作

- **LDA关键步骤**：

> 1. 对d维数据进行标准化处理（d为特征数量）
> 2. 对于每一类别，计算d维的均值向量
> 3. 构造类间的**散布矩阵** $S_B$ 以及 **类内散布矩阵** $S_W$
> 4. 计算矩阵 $S_W^{-1} S_B$ 的特征值以及对应的特征向量
> 5. 选取前k个特征值所对应的特征向量，构造一个 $d*k$ 维的转换矩阵 $W$,其中特征向量以列的形式排列
> 6. 使用转换矩阵 $W$ 将样本映射到新的特征子空间上

- 若将 $W$ 视为一个投影矩阵，则多分类LDA将样本投影到 $d'$ 维空间（$d'<<d$），于是达到了降维的目的
- 在投影过程中用到了类别信息

### 2.3 示例

- 加载数据集

```pyhton
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
```

- 划分数据集

```py
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
```

- 标准化

```py
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
```

- 计算均值向量 

$$m_i = \frac{1}{n_i}\sum_{x \in D_i}^{c}x_m$$

$$m_i = \begin{bmatrix}
\mu_{i,alcohol}\\ 
\mu_{i,malic acid}\\ 
...\\ 
\mu_{i,profilr}
\end{bmatrix}$$

```py
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))


MV 1: [ 0.9259 -0.3091  0.2592 -0.7989  0.3039  0.9608  1.0515 -0.6306  0.5354
  0.2209  0.4855  0.798   1.2017]

MV 2: [-0.8727 -0.3854 -0.4437  0.2481 -0.2409 -0.1059  0.0187 -0.0164  0.1095
 -0.8796  0.4392  0.2776 -0.7016]

MV 3: [ 0.1637  0.8929  0.3249  0.5658 -0.01   -0.9499 -1.228   0.7436 -0.7652
  0.979  -1.1698 -1.3007 -0.3912]

```

- 计算 **类内散布矩阵** $S_W$

$$S_w=\sum_{i=1}^{c}S_i $$
$$S_i=\sum_{x\in D_i}^{c}(x-m_i)(x-m_i)^T  $$

```py
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# Within-class scatter matrix: 13x13
```

- 计算**散布矩阵** $S_B$，其中$m$是全局均值，在计算时用到了所有类别中的全部样本

$$S_B=\sum_{i=1}^{c}N_i(m-m_i)(m-m_i)^T  $$

```py
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
```

- 求解矩阵 $S_W^{-1} S_B$ 的广义特征值

```py
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
```

- 接着按照降序对特征值进行排序

```py
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

Eigenvalues in decreasing order:

452.721581245
156.43636122
1.05646703435e-13
3.99641853702e-14
3.40923565291e-14
2.84217094304e-14
1.4793035293e-14
1.4793035293e-14
1.3494134504e-14
1.3494134504e-14
6.49105985585e-15
6.49105985585e-15
2.65581215704e-15
```

- 为了度量LDA可以获取多少区分类别的信息，可以按照特征降序绘制出特征对线性判别信息保持程度的图像

```py
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/lda1.png', dpi=300)
plt.show()
```

![LDA重要程度](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_07.png)

- 叠加两个判别能力最强的特征向量列构建转换矩阵 $W$：

```py
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

Matrix W:
 [[-0.0662 -0.3797]
 [ 0.0386 -0.2206]
 [-0.0217 -0.3816]
 [ 0.184   0.3018]
 [-0.0034  0.0141]
 [ 0.2326  0.0234]
 [-0.7747  0.1869]
 [-0.0811  0.0696]
 [ 0.0875  0.1796]
 [ 0.185  -0.284 ]
 [-0.066   0.2349]
 [-0.3805  0.073 ]
 [-0.3285 -0.5971]]
```

- 将样本映射到新的特征空间

```py
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0] * (-1),
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('./figures/lda2.png', dpi=300)
plt.show()
```

- 可以看到，三个葡萄酒类在新的特征子空间上是线性可分的

![LDA结果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_08.png)

### 2.4 使用sklearn进行LDA分析

- 调用sklearn中的LDA,使用Logistic回归模型：

```py
if Version(sklearn_version) < '0.18':
    from sklearn.lda import LDA
else:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./images/lda3.png', dpi=300)
plt.show()

```

- 可以看到只有两个样本被误分类。可以通过正则化对决策边界进行调整。

![LDA&Logistic](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_09.png)

- 在测试集上的表现

```py
X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./images/lda4.png', dpi=300)
plt.show()
```

![效果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_10.png)

## 3. 核主成份分析 Kernel Principle Component Analysis

### 3.1 简单介绍

- 现实世界中，并不是所有数据都是线性可分的
- 通过LDA，PCA将其转化为线性问题并不是好的方法

- **线性可分 VS 非线性可分**

![线性可分VS非线性可分](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_11.png)

- 引入**核主成份分析**

- 可以通过kPCA将非线性数据映射到高维空间，在高维空间下使用标准PCA将其映射到另一个低维空间

![kPCA效果](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/kPCA.png)

### 3.2 原理

- 定义**非线性映射函数**，该函数可以对原始特征进行非线性组合，以将原始的d维数据集映射到更高维的k维特征空间。

$$\phi :R^d\to R^k(k>>d)$$

- 例如：

$$ x=[x_1,x_2]^T\overset{\phi }{\rightarrow}z=[x_1^2,\sqrt{2x_1,x_2},x_2^2]$$

- 计算协方差矩阵 $\Sigma$ 的通用公式:

$$\Sigma = \frac{1}{n-1} XX^T$$

- 加上核函数后，计算协方差矩阵 $\Sigma$ 的通用公式:

$$\Sigma = \frac{1}{n-1} \phi(X) \phi(X)^T$$

- 常用核函数：

1. 多项式核： $\kappa (x^{(i)}, x^{(j)})=(x^{(i)T}x^{(j)}+\theta)^p$
2. 双曲正切核（sigmoid）：$\kappa (x^{(i)}, x^{(j)})=thah(\eta x^{(i)T}x^{(j)}+\theta)$
3. 径向基核（RBF），高斯核函数：$\kappa (x^{(i)}, x^{(j)})=exp(-\frac{\begin{Vmatrix}x^{(i)}-x^{(j)}\end{Vmatrix}}{2\sigma ^2})$ 或 $\kappa (x^{(i)}, x^{(j)})=exp(-\gamma \begin{Vmatrix}x^{(i)}- x^{(j)}\end{Vmatrix}^2)$

- 基于RBF核的kPCA算法流程：

>1. 计算核矩阵 $k$，做如下计算：$\kappa (x^{(i)}, x^{(j)})=exp(-\gamma \begin{Vmatrix}x^{(i)}- x^{(j)}\end{Vmatrix}^2)$。需要计算任意两样本之间的值。例如，如果数据集包含100个训练样本，将得到一个100*100维的对称核矩阵。

![eq](http://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;\kappa&space;(x^{(1)},&space;x^{(1)})&space;&\kappa&space;(x^{(1)},&space;x^{(2)})&space;&...&space;&\kappa&space;(x^{(1)},&space;x^{(n)})&space;\\&space;\kappa&space;(x^{(2)},&space;x^{(1)})&\kappa&space;(x^{(2)},&space;x^{(j)})&space;&...&space;&\kappa&space;(x^{(3)},&space;x^{(n)})&space;\\&space;...&space;&...&space;&...&space;&...&space;\\&space;\kappa&space;(x^{(n)},&space;x^{(1)})&space;&\kappa&space;(x^{(n)},&space;x^{(2)})&space;&...&space;&\kappa&space;(x^{(n)},&space;x^{(n)})&space;\end{bmatrix})

>2. 通过如下公式计算，使得核矩阵 $k$ 更为聚集：$K'=K-l_nK-Kl_n+l_nKl_n$,其中，$l_n$是一个$n*n$的矩阵，其所有的值均是1/n。
>3. 将聚集后的核矩阵的特征值按照降序排列，选择前k个特征值所对应的特征与标准PCA不同，这里的特征向量不是主成份轴，而是将样本映射到这些轴上。

### 3.3 使用Python实现kPCA

```py
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from numpy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset   

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.linalg.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))

    return X_pc

```

### 3.4 示例一

- 创建如下数据：

![data](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_12.png)

```py
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('./figures/half_moon_1.png', dpi=300)
plt.show()
```

- 直接用PCA，得到如下右图，线性不可分！

![data](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_13.png)

```py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('./figures/half_moon_2.png', dpi=300)
plt.show()
```

- 使用kPCA，使得数据线性可分

![data](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch05/images/05_14.png)

```py
from matplotlib.ticker import FormatStrFormatter

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
            color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, 
            color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
            color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

plt.tight_layout()
# plt.savefig('./figures/half_moon_3.png', dpi=300)
plt.show()
```

### 3.5 使用sklearn进行核主成份分析

- 还是调包简单！可以通过kernal参数来选择不同核函数,得到的结果与上面的左图是一致的。

```py
rom sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
# plt.savefig('./figures/scikit_kpca.png', dpi=300)
plt.show()
```

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!