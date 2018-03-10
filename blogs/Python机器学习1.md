# Python与机器学习读书笔记（一）

关于本书: [《Python机器学习》][0]

- **机器学习**：通过对自学习算法的开发，从数据中获取知识，进而对未来进行预测。
- 与以往通过大量数据分析而人工推导出规则并构建模型不同，机器学习提供了一种从数据中获取知识的方法，同时能够逐步提高预测模型的性能，并将模型应用于基于数据驱动的决策当中。

![机器学习][1]

## 1.机器学习的三种不同方法

### 1.1 监督学习 supervised learning

- **目的**：使用**有类标**的训练数据构建模型，使用该模型对未来数据进行预测。
- **监督(supervised)**：训练数据集中每个样本均有一个已知的输出项。

#### 1.1.1 分类 Classification

- 针对**离散的、无序的**输出变量进行预测
- **目的**：基于对过往类标已知示例的观察与学习，实现对新样本类标的预测。
- **例如**：下图是具有30个训练样本的实例，15个样本被标记为* 负类别(negative cycle)*  ；15个样本被标记为 *正类别(positive cycle)*。现在我们可以通过有监督的机器学习算法获得一条规则，并将其表示为一条黑色虚线标识的分界线、它可以将两类样本分开，并且可以根据给定的$x_1, x_2$值将新样本划分到某个类别中

![分类][2]

#### 1.1.2 回归 Regression

- 针对**连续型**输出变量进行预测，也就是所谓的*回归分析(regression analysis)*
- **目的**：在回归分析中，数据中会给出大量的自变量（解释变量）和相应的连续因变量（输出结果），通过尝试寻找这两种变量之间的关系，就能够预测输出变量。
- **例如**:下图阐述了*线性回归(linear regression)*的基本概念。给定一个自变量x和因变量y，拟合一条直线使得样例数据点与拟合直线之间的距离最短。这样我们就可以通过对样本数据的训练来获得拟合直线的截距和斜率，从而对新的输入变量值所对应的输出变量值进行预测。

![回归][3]

### 1.2 强化学习 reinforcement learning

- **目标**：构建一个 *系统(Agent)*，在与 *环境(environment)* 交互的过程中提高系统的性能，环境的当前状态中通常包含一个*反馈(reward)*信号。
- 我们可以将强化学习视为与监督学习相关的一个领域，然而，在强化学习中，这个反馈值不是一个确定的类标或连续类型的值，而是一个通过反馈函数产生的对当前系统行为的评价。
- 例如：象棋游戏中，Agent根据当前局势（environment）决定落子位置。

![强化学习][4]

### 1.3 无监督学习 unsupervised learning

- **处理对象**：无类标数据或总体分布趋势不明朗的数据。
- 可以在没有已知输出变量和反馈函数指导的情况下提取有效信息来探索数据的整体结构。

#### 1.3.1 通过聚类发现数据的子群

- **聚类(cluster)**：一种探索性数据分析技术，在没有任何相关先验信息的情况下，它可以帮助我们将数据划分为有意义的小的组别(即 簇 cluster)。
- 对数据分析时，生成的每个簇中其内部成员之间具有一定的相似度，而与其它簇中的成员则具有较大的不同。
- 例如：下图演示了聚类方法如何根据数据的$x_1, x_2$两个特征之间的相似性将无类别的数据划分到三个不同的组中。

![聚类][5]

#### 1.3.2 数据压缩中的降维

- 通常，我们面对的数据都是高维的，这就对有限的数据存储空间以及机器学习算法提出了挑战。
- **无监督降维**是特征预处理常用技术，用于清楚数据中的噪声，能够在最大程度保留相关信息的情况下将数据压缩到一个维度较小的子空间，但同时也可能会降低某些算法的准确性方面的性能。
- 例如：下图展示了使用降维方法将三维数据压缩到二维特征子空间的实例。

![降维][6]

## 2. 基本术语及符号介绍

下面的表格摘录了鸢尾花(Iris)数据集中的部分数据。该数据集包含了**3个品种**总共**150朵鸢尾花**的测量数据。其中，每一个样本代表数据集中的一行，而花的测量值以cm为单位存储为列，我们将其定义为数据的**特征(Feature)**。

![鸢尾花][7]

## 3. 构建机器学习的蓝图

阐述了使用机器学习的预测模型进行数据分析的典型工作流程图
![机器学习流程图][9]

### 3.1 数据预处理

- **目的**：尽可能发挥机器学习算法的性能。
- **方法**：提取有效特征、归一化、降维去除数据冗余等。
- 通常我们还会将数据集划分为**训练数据集**和**测试数据集**，前者用于训练模型，后者用于模型评估。

### 3.2 选择预测模型类型并进行训练

- 选用几种不同的算法来训练模型，并比较它们的性能，选择最优的一个。

### 3.3 模型验证与使用未知数据进行预测

- 采用测试数据对模型进行测试，评估模型的泛化能力。

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!

[0]:https://book.douban.com/subject/27000110/
[1]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/ml.png
[2]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/classification.png
[3]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/regression.png
[4]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/reinforcement%20learning.png
[5]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/cluster.png
[6]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/dimensionality%20reduction.png
[7]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/iris.png
[8]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/notations.png
[9]:https://github.com/KARL13YAN/pictures-for-my-blogs/raw/master/pythonML/ml%20detail.png