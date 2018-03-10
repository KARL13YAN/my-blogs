# Python与机器学习读书笔记（三）数据预处理

说明：

1. 关于本书: [《Python机器学习》][0]
2. 本笔记侧重代码调用，只描述了一些简单概念，本书的公式推导不在这里展示
3. 本页[代码](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/ch04.ipynb)

---

机器学习算法结果的优劣与数据的质量和数据中蕴含的有用信息量的数量息息相关，因此数据预处理事关重要。

## 1. 处理数值型数据

数据如下：

![数据][1]

```python
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
```

### 1.1 缺失值统计[dropna](http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.DataFrame.dropna.html?highlight=dropna#pandas.DataFrame.dropna)

```python
df.isnull().sum()

A    0
B    0
C    1
D    1
dtype: int64
```

### 1.2 删除存在缺失值的**样本**

`df.dropna()`

![删除存在缺失值的样本][2]

### 1.3 删除存在缺失值的**特征**

`df.dropna(axis=1)`

![删除特征][3]

- dropna的其它用法：

```python
# only drop rows where all columns are NaN
df.dropna(how='all')

        A	B	C	D
0	1.0	2.0	3.0	4.0
1	5.0	6.0	NaN	8.0
2	10.0	11.0	12.0	NaN


# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)

        A	B	C	D
0	1.0	2.0	3.0	4.0


# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

    A   B   C   D
0   1.0 2.0 3.0 4.0
2   10.0    11.0  12.0  NaN
```

### 1.4 **填充**缺失值

这里介绍的是**均值插补(meaneinputation)**，sklearn中的[Impute](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer)类

```python
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data
```

### 1.5 理解sklearn中的预估器API

sklean中的Impute类属于sciait-learn中的转换器类，主要用于数据的转换，常用的两个方法为：

- **fit方法**用于对数据集中的参数进行识别并构建相应的数据补齐模型
- **transform方法**使用刚构建的数据补齐模型对数据集中的相应参数缺失值进行补齐

![预估器模型](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_04.png)

## 2. 处理类别数据

在真实数据集中通常会出现一个或多个**类别数据**的特征列。进一步，可以将类别数据划分为**标称特征（nominal feature）** 和 **有序特征（ordinal feature）**

- 有序特征：类别的值是有序或可以排序的
- 标称特征：不具备排序的特性

数据集如下：

![数据集](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_06.png)

```python

import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
```

### 2.1 类标的编码

sklearn的[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder)类

```python
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# y:array([0, 1, 0])

#还原
class_le.inverse_transform(y)
# array(['class1', 'class2', 'class1'], dtype=object)
```

### 2.2 有序特征的映射

把类别字符串转化为整数，需要手动定义

```python
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
```

结果![结果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_07.png)

### 2.3 标称特征的独热编码

比如：将color特征转化为三个新特征：blue、green、red

- sklearn [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()
```

- pandas [get_dummies](http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.get_dummies.html?highlight=dummies#pandas.get_dummies)

```python
pd.get_dummies(df[['price', 'color', 'size']])
```

![结果](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_09.png)

## 3. 数据集划分

将数据集划分为训练集和测试集

```python
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
```

## 4. 特征缩放

- 决策树和随机森林**不需要**特征缩放

### 4.1 归一化(normalization)

将特征缩放到区间[0,1]，是最小-最大缩放的一个特例。计算公式如下：

![公式](http://latex.codecogs.com/gif.latex?x_{norm}^{(i)}=\frac{x^{(i)}-x_{min}}{x_{max}-x_{min}})

- sklearn [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
```

### 4.2 标准化(standardization)

可以将特征列的均值设为0，方差为1，是的特征列的值呈标准正态发布，更易于权重更新，保持了异常值所蕴含的有用信息，公式如下

![正态化](http://latex.codecogs.com/gif.latex?x_{std}^{(i)}=\frac{x^{(i)}-\mu_{x}}{\sigma_x})

- sklearn [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```

### 4.3 标准化 VS 归一化

![标准化 VS 归一化](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/std%20vs%20norm.png)

```python
x = pd.DataFrame([0, 1, 2, 3, 4, 5])

# standardize
ex[1] = (ex[0] - ex[0].mean()) / ex[0].std(ddof=0)

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
ex[2] = (ex[0] - ex[0].min()) / (ex[0].max() - ex[0].min())
ex.columns = ['input', 'standardized', 'normalized']
ex
```

## 5 通过降低模型复杂度防止过拟合

降低泛化误差的方案：

1. 收集更多的训练数据
2. **通过正则化引入罚项**
3. 选择一个参数较少的简单模型
4. **降低数据维度**

### 5.1 使用L1正则化满足数据稀疏化

待补充

### 5.2 序列特征选择算法

- 降维主要分为两大类：

1. 特征抽取：将原始数据变换到一个维度更低的新的特征子空间
2. 特征选择：选出原始特征的一个子集

- **序列特征选择算法**是一种贪婪搜索算法，用于将原始的 $d$ 维特征空间压缩到一个 $k$ 维特征子空间，其中 $k<d$ 。能够剔除不相关特征或噪声，自动选出与问题最相关的特征子集，从而提高计算效率或是降低模型的泛化误差。
- 一个经典的序列特征选择算法，**序列后向选择算法（Sequential Backward Selection， SBS）**，其目的是在分类性能衰减最小的约束下，降低原始特征空间上的数据维度以提高计算效率，可以在模型面临过拟合问题时提高数据的预测能力。
- SBS算法理念：SBS依此从特征集合中删除某些特征，直到新的特征子空间包含指定数量的特征。定义个一个需要最小化的衡量函数$J$，比较判定分类器的性能在删除某个特定特征前后的差异，也就是说，每一个删除能使$J$尽可能大的特征。

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!

[0]:https://book.douban.com/subject/27000110/
[1]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_01.png
[2]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_02.png
[3]:https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch04/images/04_03.png