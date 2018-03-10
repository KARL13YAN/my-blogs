# Python与机器学习读书笔记（二）分类算法及sklearn调用

说明：

1. 关于本书: [《Python机器学习》][0]
2. 本笔记侧重代码调用，只描述了一些简单概念，本书的公式推导不在这里展示

## 1. sklearn基础

### 1.1 调用鸢尾花数据集

提取鸢尾花数据集的两个特征

```Python
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
```

### 1.2 数据集划分

```python
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
```

### 1.3 标准化处理

fit后可以获得每个特征的样本均值和样本标准差，来对数据进行标准化处理

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

### 1.4 分类准确率

```python
from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

### 1.5 绘制模型决策区域

```python
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
```

## 2 感知机 Perceptron

### 2.1 感知机原理

![感知机][1]

感知机接收样本x的输入，并将其与权值w进行加权得到**净输入(net inout)**，接着传递到**激励函数（activation function）**生成值为+1或-1的二值输出，并将该输出作为样本的预测类标，在学习阶段，该输出用来计算预测的误差并更新权重。

- 在样本不是完全线性可分的情况下，永远不收敛

### 2.2 sklearn

```python
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 4

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()
```

![感知机运行结果][3]

## 3 自适应线性神经元 Adaptive Linear Neuron

![自适应线性神经元][2]

与感知机不同的地方在于，这里的**激励函数（activation function）** 是线性的:$\phi (z)=z$，而在激励函数与输出之间增加了**量化器（quantizer）**对类标进行预测，量化器类似于单位阶跃函数。

## 4. Logistic回归

### 4.1 原理

感知机在样本不是完全线性可分的情况下，永远不收敛，因此提出了Logistic回归方法，该方法是**分类模型**

- 几率比 $\frac{p}{1-p}$
- 几率比的对数函数即logit函数 $logit(p)=log\frac{p}{1-p}$
- 对数几率作为输入特征值表达式 $log\frac{p}{1-p} =w^Tx$
- Sigmoid函数，即logit函数的反函数 $\phi (z)=\frac{1}{1+e^{-z}}$ ,，其中$z=w^Tx$，函数图像如下
- Sigmoid函数以实数作为输入并将其映射到$[0,1]$区间，即在给定特征x及其权重w的情况下，该函数能给出x属于类别1的概率 $\phi (z)=P(y=1|x;w)$

![sigmoid函数][4]

其过程如下图所示。相比于自适应线性神经元，此处的激励函数是sigmoid函数。

![logistic回归][5]

### 4.2 sklearn 

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
plt.show()
```

可以预测样本属于某一类别的概率

```pyhton
if Version(sklearn_version) < '0.17':
    lr.predict_proba(X_test_std[0, :])
else:
    lr.predict_proba(X_test_std[0, :].reshape(1, -1))
```

![使用logistic回归处理鸢尾花数据集][6]

## 5. 支持向量机 Support Vector Machine, SVM

### 5.1 原理

- 可以看作是感知机的扩展
- 感知机中，**最小化分类误差**；SVM中，**最大化分类间隔**
- **间隔**：决策边界之间的距离
- **支持向量**：最靠近超平面的训练样本
- 对于非线性可分的数据，引入松弛变量 $\xi$，放松线性约束条件，保证在适当的惩罚成本下，对错误分类的情况进行优化时能够收敛

![SVM][7]

- 变量C

![变量C][8]

### 5.2 sklearn

``` pyhton
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()
```

![SVM处理鸢尾花数据集][9]

### 5.3 核SVM

-**核方法**：通过映射函数$\phi(.)$将样本的原始特征映射到一个使样本线性可分的更高维空间中。

![核方法][10]

### 5.4 sklearn

```python
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
plt.show()
```

![处理异或数据][11]

下面是调整参数 gamma 对模型的影响，第一张图，gamma=0.2，第二张图, gamma=100

![gamma=0.2][12]
![gamma=100][13]

## 6. 决策树 Decision Tree

![决策树][14]

### 6.1 原理

- 基于可获得最大**信息增益（information gain**的特征来对数据进行划分
- **信息增益**只不过是父节点的不纯度与所有子节点的不纯度和之差——子节点不纯度越低，信息增益越大
- 常用的3个不纯度衡量标准：**基尼系数（Gini Index）**，**熵（entropy）**，**误分类率（Classification error）**

![三种不同的不纯度衡量标准][15]

### 6.2 sklearn

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/decision_tree_decision.png', dpi=300)
plt.show()
```

![决策树处理鸢尾花数据集][16]

- sklean支持将训练后的决策树导出为.dot的格式，用GraphViz可视化

## 7. k-近邻算法 k-nearest neighbor classifier, KNN

### 7.1 原理

- 非参数化模型
- 算法步骤：
1. 选择近邻数量k和距离度量方法
2. 找到待分类样本的k个最近的邻居
3. 根据最近邻的类标进行多数投票

![KNN][17]

### 7.2 sklearn

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
plt.show()
```

![KNN处理鸢尾花数据集][18]

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!

[0]:https://book.douban.com/subject/27000110/
[1]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/perceptron.png
[2]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/Adaptive%20Linear%20Neuron.png
[3]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_01.png
[4]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_02.png
[5]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_03.png
[6]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_05.png
[7]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_07.png
[8]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_08.png
[9]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_09.png
[10]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_11.png
[11]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_12.png
[12]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_13.png
[13]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_14.png
[14]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_15.png
[15]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_16.png
[16]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_17.png
[17]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_20.png
[18]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_21.png