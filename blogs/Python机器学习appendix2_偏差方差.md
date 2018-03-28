# Python机器学习读书笔记（十一）——偏差&方差

> 对学习算法除了通过实验估计其泛化性能，人们往往还希望了解它“为什么”具有这样的性能。“偏差-方差分解”是**解释学习算法泛化性能**的一种重要工具。（周志华 《机器学习》）

## **偏差&方差的引入**

- 泛化误差可以理解为偏差、方差与噪声之和

  - $$E(f;D)=bias^2(x)+var(x)+\epsilon^2$$

- **偏差**度量了算法预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力。
- **方差**度量了同样大小的训练接的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响。
- **噪声**表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，即刻画了学习问题本身的难度。

- 偏差-方差分解说明，**泛化性能由算法的学习能力、数据的充分性以及学习任务本身难度所共同决定的。**

- 良好的泛化性能 ，意味着：能够充分拟合数据（偏差较小），数据扰动产生的影响较小（方差较小）

- 一般来说，偏差和方差是有冲突的，称之为偏差-方差囧境（bias-variance dilemma），如下图所示，截图来自周志华 《机器学习》

![bias_variance](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/bias_variance.png)

- 此外，吴恩达在他的机器学习课程上，对偏差-方差也有形象的阐述，如下图所示。

![bias_ng](http://mmbiz.qpic.cn/mmbiz_jpg/BIZeGU4QG999fC2o1W2ibmhOabeicutTbfmq6el16cXTibib1Drs8yx2fXDFy71W54qQkAxlia7Q8xicsVQWjfZBUPQg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

- 总的来说，上面两张图阐述了这样的思想：偏差减小的代价，是方差随之增大，并产生过拟合；方差减小的代价，是偏差随之增大，并产生欠拟合

- 在偏差与方差间寻找平衡点，是一个重要的工程。

## **举个例子**

假设模型是一个射击学习者，$D_1$,$D_2$直到$D_N$就是N个独立的训练计划。

如果一个学习者是正常人，一个眼睛斜视，则可以想见，斜视者无论参加多少训练计划，都不会打中靶心，问题不在训练计划够不够好，而在他的先天缺陷。这就是模型偏差产生的原因，学习能力不够。正常人参加N个训练计划后，虽然也不能保证打中靶心，但随着N的增大，会越来越接近靶心。

假设还有一个超级学习者，他的学习能力特别强，参加训练计划D1时，他不仅学会了瞄准靶心，还敏感地捕捉到了训练时的风速，光线，并据此调整了瞄准的方向，此时，他的训练成绩会很好。

但是，当参加测试时的光线，风速肯定与他训练时是不一样的，他仍然按照训练时学来的瞄准方法去打靶，肯定是打不好。这样产生的误差就是方差。这叫做聪明反被聪明误。

当我们只有一份训练数据D时，我们选的M若太强，好比射手考虑太多风速，光线等因素，学出来的模型Mt在测试样本上表现肯定不好，若选择的M太挫，比如是斜视，也无论如何在测试的样本上表现也不会好。所以，最好的M就是学习能力刚刚好的射手，它能够刚刚好学习到瞄准的基本办法，又不会画蛇添足地学习太多细枝末节的东西。

## **通过学习曲线判定偏差与方差问题**

![引子](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch06/images/06_04.png)

- 左上图，是一个**高偏差**模型。训练准确率和交叉验证准确率都偏低，表明该模型**欠拟合**

  - 可以增加模型的参数数量，或降低正则化程度来解决该问题

- 右上图，是一个**高方差**模型。训练准确率和交叉验证准确率之间有较大差距

  - 可以收集更多训练数据或降低模型复杂度（如使用正则化方法）解决问题

- 右下图，即是**偏差-方差平衡**

### **使用sklearn绘制验证曲线**

```py
import matplotlib.pyplot as plt

if Version(sklearn_version) < '0.18':
    from sklearn.learning_curve import learning_curve
else:
    from sklearn.model_selection import learning_curve



pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2', random_state=0))])

train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()
```

- 可得如下曲线

![曲线](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_05.png)

- `learning_curve`函数的`train_size`参数，可以控制用于生成学习曲线的样本的绝对或相对数量，在这里，`train_sizes=np.linspace(0.1, 1.0, 10)`

- 默认情况下，`learning_curve`函数使用**分层k折交叉验证**来计算交叉验证的准确性，这里`k=10`

- 由图可见，模型在测试数据集上表现良好，但是**准确率曲线和交叉验证曲线存在较小差距**，意味着存在**轻微过拟合**

## **通过验证曲线判定拟合与过拟合**

- 验证曲线是一种通过定位欠拟合或过拟合等问题的所在，来帮助提高模型性能的有用工具。

- 描述的是**准确率与模型参数之间的关系**

### ** 使用sklearn绘制验证曲线**

```py
if Version(sklearn_version) < '0.18':
    from sklearn.learning_curve import validation_curve
else:
    from sklearn.model_selection import validation_curve



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr,
                X=X_train,
                y=y_train,
                param_name='clf__C',
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('./figures/validation_curve.png', dpi=300)
plt.show()
```

![结果](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_06.png)

- `validation_curve`默认使用**分层k折交叉验证**
- 在该函数中，**可以指定想要验证的参数**，本例中所要验证的参数是`C`，`param_name='clf__C',param_range=param_range,`前一句将C记为`clf__C`，并通过`param_range`参数设定其范围

- 可以看到，加大正则化强度（较小的C）会导致轻微欠拟合；增加C值，以为只降低正则化强度，模型趋于过拟合
- 本例中，C=0.1是最优点

## 下期预告

后面的文章会带来**正则化以及网格搜索调优**的内容

---
**参考**：

1. [【原理】机器学习偏差与方差](http://mp.weixin.qq.com/s/rV8OEKxTcaC_5rVau7ta2w)
2. [机器学习](https://book.douban.com/subject/26708119/)

---

作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!