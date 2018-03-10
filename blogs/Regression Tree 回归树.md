# Regression Tree 回归树

---

## 1. 引言

AI时代，机器学习算法成为了研究、应用的热点。当前，最火的两类算法莫过于**神经网络算法**（CNN、RNN、LSTM等）与**树形算法**（随机森林、GBDT、XGBoost等），树形算法的基础就是**决策树**。决策树因其易理解、易构建、速度快的特性，被广泛应用于统计学、数据挖掘、机器学习领域。因此，对决策树的学习，是机器学习之路必不可少的一步。

根据处理数据类型的不同，决策树又分为两类：**分类决策树**与**回归决策树**，前者可用于处理离散型数据，后者可用于处理连续型数据，下面的英文引用自[维基百科][1]。

> **Classification tree** analysis is when the predicted outcome is the class to which the data belongs.
>
> **Regression tree** analysis is when the predicted outcome can be considered a real number (e.g. the price of a house, or a patient's length of stay in a hospital).

网络上有关于分类决策树的介绍可谓数不胜数，但是对回归决策树（回归树）的介绍却少之又少。李航教授的[统计学习方法][2] 对回归树有一个简单介绍，可惜篇幅较短，没有给出一个具体实例；Google搜索*回归树*，有[一篇介绍回归树的博客（点击）][3]，该博客所举的实例**有误**，其过程事实上是**基于残差的GBDT**。

基于以上原因，本文简单介绍了回归树（Regression Tree），简单描述了CART算法，给出了回归树的算法描述，辅以简单实例以加深理解，最后是总结部分。

## 2. 回归树

决策树实际上是将空间用超平面进行划分的一种方法，每次分割的时候，都将当前的空间一分为二， 这样使得每一个叶子节点都是在空间中的一个不相交的区域，在进行决策的时候，会根据输入样本每一维feature的值，一步一步往下，最后使得样本落入N个区域中的一个（假设有N个叶子节点），如下图所示。

![决策树][4]

三种比较常见的**分类决策树**分支划分方式包括：ID3, C4.5, CART。

![分类决策树][5]

分类与回归树（classificationandregressiontree, CART)模型由Breiman等人在1984年提出，是应用广泛的决策树学习方法。CART同样由特征选择、树的生成及剪枝组成，既可以用于分类也可以用于回归。下面的英文引用自[维基百科][1]

>The term **Classification And Regression Tree (CART)** analysis is an umbrella term used to refer to both of the above procedures, first introduced by Breiman et al. Trees used for regression and trees used for classification have some similarities - but also some differences, such as the procedure used to determine where to split.

下面介绍回归树。

### 2.1 原理概述

既然是决策树，那么必然会存在以下两个核心问题：**如何选择划分点？如何决定叶节点的输出值？**

一个回归树对应着输入空间（即特征空间）的一个划分以及在划分单元上的输出值。**分类树**中，我们采用信息论中的方法，通过计算选择最佳划分点。而在**回归树**中，采用的是启发式的方法。假如我们有n个特征，每个特征有$s_i(i \in (1,n))$个取值，那我们遍历所有特征，尝试该特征所有取值，对空间进行划分，直到取到特征j的取值s，使得损失函数最小，这样就得到了一个划分点。描述该过程的公式如下：

![损失][7]

假设将输入空间划分为M个单元：$R_1,R_2,...,R_m$ 那么每个区域的输出值就是：$c_m=ave(y_i|x_i \in R_m)$也就是该区域内所有点y值的平均数。

举个例子。如下图所示，假如我们想要对楼内居民的年龄进行回归，将楼划分为3个区域$R_1, R_2, R_3$（红线），那么$R_1$的输出就是第一列四个居民年龄的平均值，$R_2$的输出就是第二列四个居民年龄的平均值，$R_3$的输出就是第三、四列八个居民年龄的平均值。
![一个例子][6]

### 2.2 算法描述

截图来自李航教授的[统计学习方法][2]

![CART算法描述][8]

### 2.3 一个简单实例

为了便于理解，下面举一个简单实例。训练数据见下表，目标是得到一棵最小二乘回归树。

| x | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| - | - | - | - | - | - | - | - | - | - | -  |
| y | 5.56 | 5.7 | 5.91 | 6.4| 6.8 | 7.05 | 8.9 | 8.7 | 9 | 9.05 |

1. 选择最优切分变量j与最优切分点s

    在本数据集中，只有一个变量，因此最优切分变量自然是x。

    接下来我们考虑9个切分点$[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]$你可能会问，为什么会带小数点呢？类比于篮球比赛的博彩，倘若两队比分是96:95，而盘口是“让1分 A队胜B队”，那A队让1分之后，到底是A队赢还是B队赢了？所以我们经常可以看到“让0.5分 A队胜B队”这样的盘口。在这个实例中，也是这个道理。

    损失函数定义为平方损失函数 $Loss(y, f(x))=(f(x)-y)^2$，将上述9个切分点一依此代入下面的公式，其中 $c_m=ave(y_i|x_i \in R_m)$

    ![损失][7]

    例如，取 $s=1.5$。此时
    $R_1=\{1\} , R_2=\{2,3,4,5,6,7,8,9,10\}$，这两个区域的输出值分别为：$c_1=5.56, c_2= \frac{1}{9}(5.7+5.91+6.4+6.8+7.05+8.9+8.7+9+9.05)=7.50$。得到下表：

    | s | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
    | - | - | - | - | - | - | - | - | - | - | -  |
    | $c_1$ | 5.56 | 5.63 | 5.72 | 5.89 | 6.07 | 6.24 | 6.62 | 6.88 | 7.11 |
    | $c_2$ | 7.5 | 7.73 | 7.99 | 8.25 | 8.54 | 8.91 | 8.92 | 9.03 | 9.05 |

    把$c_1, c_2$的值代入到上式，如：$m(1.5)=0+15.72=15.72$。同理，可获得下表：
    | s | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 |
    | - | - | - | - | - | - | - | - | - | - | -  |
    | m(s) | 15.72 | 12.07 | 8.36 | 5.78| 3.91 | 1.93| 8.01 | 11.73 | 15.74 |
    显然取 $s=6.5$时，$m(s)$最小。因此，第一个划分变量$j=x, s=6.5$

2. 用选定的(j,s)划分区域，并决定输出值

    两个区域分别是：$R_1=\{1,2,3,4,5,6\} , R_2=\{7,8,9,10\}$输出值$c_m=ave(y_i|x_i \in R_m),c_1=6.24,c_2=8.91$
3. 对两个子区域继续调用步骤1、步骤2

    对$R_1$继续进行划分:

    | x | 1 | 2 | 3 | 4 | 5 | 6 |
    | - | - | - | - | - | - | - |
    | y | 5.56 | 5.7 | 5.91 | 6.4| 6.8 | 7.05|
    取切分点$[1.5, 2.5, 3.5, 4.5, 5.5]$，则各区域的输出值$c$如下表
    | s | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 |
    | - | - | - | - | - | - | - | - | - | - | -  |
    | $c_1$ |5.56 | 5.63 | 5.72 | 5.89 | 6.07|
    | $c_2$ |6.37 | 6.54 | 6.75 | 6.93 | 7.05|
    计算m(s)：
    | s | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 |
    | - | - | - | - | - | - | - | - | - | - | -  |
    | m(s) |1.3087 | 0.754 | 0.2771 | 0.4368 | 1.0644|

    s=3.5时m(s)最小。

    **之后的过程不再赘述。**

4. 生成回归树

    **假设**在生成**3个区域**之后停止划分，那么最终生成的回归树形式如下：

    $T=\left\{\begin{matrix}5.72 & x\leq 3.5\\ 6.75 &3.5\leqslant x\leq 6.5\\ 8.91 & 6.5<x\end{matrix}\right.$

### 2.4 回归树VS线性回归

不多说了，直接看图甩[代码][9]

![回归树VS线性回归][10]

## 3. 总结

实际上，回归树总体流程类似于分类树，分枝时穷举每一个特征的每一个阈值，来寻找最优切分特征j和最优切分点s，衡量的方法是平方误差最小化。分枝直到达到预设的终止条件(如叶子个数上限)就停止。

当然，处理具体问题时，单一的回归树肯定是不够用的。可以利用集成学习中的boosting框架，对回归树进行改良升级，得到的新模型就是提升树（Boosting Decision Tree），在进一步，可以得到梯度提升树（Gradient Boosting Decision Tree，GBDT），再进一步可以升级到XGBoost。

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!

[1]:https://en.wikipedia.org/wiki/Decision_tree_learning#Decision_tree_types
[2]:https://book.douban.com/subject/10590856/
[3]:https://cethik.vip/2016/09/21/machineCAST/
[4]:https://github.com/KARL13YAN/learning/raw/master/pic/1%20dt%20divide.png
[5]:https://github.com/KARL13YAN/learning/raw/master/pic/1%20dt%20type.png
[6]:https://github.com/KARL13YAN/learning/raw/master/pic/1%20dt%20example.png
[7]:http://latex.codecogs.com/gif.latex?\min_j_s[&space;\min_c___1&space;Loss(y_i,&space;c_1)&space;&plus;&space;\min_c___2&space;Loss(y_i,&space;c_2)&space;]
[8]:https://github.com/KARL13YAN/learning/raw/master/pic/1%20dt%20cart.png
[9]:https://github.com/KARL13YAN/learning/blob/master/regression%20tree.py
[10]:https://github.com/KARL13YAN/learning/raw/master/pic/1%20dt%20sklearn.png