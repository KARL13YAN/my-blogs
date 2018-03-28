# Python机器学习读书笔记（六）模型评估与参数调优

---

## 1. **基于流水线的工作流**

> sklearn中的`Pipline`类，可以拟合出包含任意多个处理步骤的模型，并将模型用于新数据的预测

- 威斯康辛乳腺癌数据集（Breast Cancer Wisconsin）：569个样本，第1列是样本ID，第二列是肿瘤诊断结果（M代表恶性，B代表良性），后面的30列均是特征。

- 使用pandas读取数据集

```py
import pandas as pd
import urllib

try:
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                     '/breast-cancer-wisconsin/wdbc.data', header=None)

except urllib.error.URLError:
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                     'python-machine-learning-book/master/code/'
                     'datasets/wdbc/wdbc.data', header=None)
print('rows, columns:', df.shape)
df.head()
```

- 将数据集的30个特征赋值给Numpy数组对象；使用sklearn中的LabelEncoder类将类标从原始字符串转化成整数

```py
from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.transform(['M', 'B']) # array([1, 0])
```

- 划分数据集

```py
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=1)
```

- 出于性能优化的目的，需要对特征做标准化处理；此外需要用PCA将数据压缩到二维子空间上；最后用Logistic回归进行分类。
- `Pipline`对象采用元组的序列作为输入。**第一个值是字符串**，可以是任意标识符，通过它来访问流水线中的元素；**第二个值是sklearn中的一个转换器或评估器**。

```py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
y_pred = pipe_lr.predict(X_test) # Test Accuracy: 0.947
```

- 上述代码的流水线有三个流程：**数据标准化StandardScaler**、 **数据降维PCA**、 **评估器LogisticRegression**。当`pipe_lr.fit`执行`fit`操作时，``StandardScaler`以及`PCA`会在训练数据上执行`fit`和`transform`操作，并将处理过的数据传递到流水线的最后一个对象`LogisticRegression`。流水线工作方式可以用下图来描述。

![pipline](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_01.png)

## 2. **交叉验证**

- 构建机器学习模型的一个关键步骤，就是在新数据上对模型的性能进行评估
- 如果模型**过于简单**，将会面临**欠拟合（高偏差）**的问题
- 如果模型基于训练数据构造的**过于复杂**，则会导致**过拟合（高方差）**
- 为了在偏差和方差之间找到可接受的折中方案，需要对模型进行评估

- 下面介绍两种评估方法

### 2.1 **holdout方法 holdout cross-validation**

- **最初的方法**：将数据划分为训练集和测试集
- 如果不断重复使用相同的测试数据进行调参，将会造成过过拟合

- **holdout方法**：将数据划分为**训练集**、**验证集**和**测试集**
- 训练集用于拟合模型
- 模型在验证集的表现作为模型选择标准
- 测试集用来评估模型

![holdout](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_02.png)

- 缺点：

1. 模型性能的评估对训练数据集划分为训练及验证子集的方法是敏感的
2. 评价结果会随着样本的不同而发生改变

### 2.2 **k折交叉验证 k-fold cross-validation**

- 不重复地随机将训练数据集划分为k个子数据集，其中前k-1个用于模型训练，剩余一个用于测试

- 重复以上过程k次，就得到了k个模型及对模型性能的评价
- 与holdout方法相比，这样得到的结果对数据划分方法的敏感性相对较低

- 通常情况下：

> 1. 将k折交叉验证用于模型的调优（即找到使模型最优的超参数）
> 2. 一旦找到了满意的超参数，在全部的训练数据集上重新训练模型
> 3. 使用独立的测试数据集对模型性能做出评价

![k-fold](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_03.png)

- k的选择

> 1. k的标准值是10
> 2. 如果数据集相对较小，则增大k的值
> 3. 如果数据集较大，则可以选择较小的k值，比如K=5

- k折交叉验证的一个特例：**留一交叉验证法（leave-one-out）**，此时**k=n**，每次只有一个样本用于测试，当数据集较小时，可以使用这种方法。

- sklearn中的k折交叉验证。`n_jobs`可以将不同分块的性能评估分配到多个CPU上进行，`n_jobs=-1`代表使用计算机所有的CPU进行计算。

```py
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import cross_val_score
else:
    from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

CV accuracy scores: [ 0.89130435  0.97826087  0.97826087  0.91304348  0.93478261  0.97777778
  0.93333333  0.95555556  0.97777778  0.95555556]
CV accuracy: 0.950 +/- 0.029
```

### 2.3 **分层k折交叉验证**

- 分层k折交叉验证是对k折交叉验证的稍许改进
- **类别比例相差较大**时，在分层交叉验证中，类别比例在每个分块中得以保持，使得每个分块中的类别比例与训练数据集的整体比例一致

```py
import numpy as np

if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import StratifiedKFold
else:
    from sklearn.model_selection import StratifiedKFold


if Version(sklearn_version) < '0.18':
    kfold = StratifiedKFold(y=y_train,
                            n_folds=10,
                            random_state=1)
else:
    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))

print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


Fold: 1, Class dist.: [256 153], Acc: 0.891
Fold: 2, Class dist.: [256 153], Acc: 0.978
Fold: 3, Class dist.: [256 153], Acc: 0.978
Fold: 4, Class dist.: [256 153], Acc: 0.913
Fold: 5, Class dist.: [256 153], Acc: 0.935
Fold: 6, Class dist.: [257 153], Acc: 0.978
Fold: 7, Class dist.: [257 153], Acc: 0.933
Fold: 8, Class dist.: [257 153], Acc: 0.956
Fold: 9, Class dist.: [257 153], Acc: 0.978
Fold: 10, Class dist.: [257 153], Acc: 0.956

CV accuracy: 0.950 +/- 0.029
```

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!