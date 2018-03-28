# Python机器学习读书笔记（十四）集成学习概述

> [前面的文章](https://blog.csdn.net/weixin_40604987)提到了不少机器学习算法，有些算法看起来有些“弱”而略显鸡肋，那么有没有办法让这些“弱”算法变“强”呢？答案就是**集成方法**

---

## 集成方法概述

- 本身不是一个单独的机器学习算法，而是通过训练多个分类器，然后把这些分类器组合起来，以达到更好的预测性能。也就是我们常说的“博采众长”

![el1](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el1.png)

- 如图所示，对于训练集数据，我们通过训练若干个个体学习器，通过一定的结合策略，就可以最终形成一个强学习器，以达到博采众长的目的

- 也就是说，集成学习有两个主要的问题需要解决
  1. 如何得到若干个个体学习器
  2. 如何选择一种结合策略，将这些个体学习器集合成一个强学习器

### 个体学习器

![el2](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el2.png)

- **同质**:个体学习器都是一个种类的，比如都是决策树个体学习器，或者都是神经网络个体学习器
- **异质**:所有的个体学习器不全是一个种类的
- 同质个体学习器按照个体学习器之间是否存在依赖关系可以分为两类：Bagging和Boosting

### 结合策略

- 结合策略通常有三种

![el3](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el3.png)

- 用Python实现了一个简单的多数投票分类器，注释比较多，就不啰嗦了。

```py
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
```

## Bagging

- **B**ootstrap **agg**regat**ing** 引导聚集算法、装袋算法
- Bagging 的个体学习器之间不存在强依赖关系，一系列个体学习器可以并行生成

![el4](https://images2015.cnblogs.com/blog/1042406/201612/1042406-20161204200000787-1988863729.png)

### 随机抽样 Bootstrap

- 为什么要随机抽样？
  - **使强学习器更健壮**如果采取某种有规律的采样，那么生成的T个采样集可能相似度极高，训练出的T个弱学习器相似度也极高，那么集成这T个弱学习器的意义就没有了。

- **Bootstrap**就是一种有放回采样方法。
  1. 假定我们想构造T个弱学习器，则需要生成T个采样集用来训练；
  2. 如果原始数据集有m个样本，则采样集也需要m个样本。

![el5](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el5.png)

![el6](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el6.png)

![el7](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el7.png)

## Boosting

- 训练过程为阶梯状，基模型按次序一一进行训练（实现上可以做到并行），基模型的训练集按照某种策略每次都进行一定的转化。对所有基模型预测的结果进行线性综合产生最终的预测结果

![el8](https://images2015.cnblogs.com/blog/1042406/201612/1042406-20161204194331365-2142863547.png)

- boosting算法的工作机制是首先从训练集用初始权重训练出一个弱学习器1，根据弱学习的学习误差率表现来更新训练样本的权重，使得之前弱学习器1学习误差率高的训练样本点的权重变高，使得这些误差率高的点在后面的弱学习器2中得到更多的重视。然后基于调整权重后的训练集来训练弱学习器2.，如此重复进行，直到弱学习器数达到事先指定的数目T，最终将这T个弱学习器通过集合策略进行整合，得到最终的强学习器

## Stacking

- 将训练好的所有基模型对训练基进行预测，第j个基模型对第i个训练样本的预测值将作为新的训练集中第i个样本的第j个特征值，最后基于新的训练集进行训练。同理，预测的过程也要先经过所有基模型的预测形成新的测试集，最后再对测试集进行预测

![el9](https://github.com/KARL13YAN/my-blogs/raw/master/pythonML/el9.png)

---

> 文中的图来自[1](http://www.cnblogs.com/pinard/p/6131423.html)、[2](https://mp.weixin.qq.com/s?__biz=MzA4OTg5NzY3NA==&mid=2649345415&idx=1&sn=cb9b7721f27dd2e80909081fa4f4f6c8&chksm=880e8222bf790b34995f207ebea7debad6e68cd7f82015cd19d8268d2dca661863450e05e5e2&mpshare=1&scene=1&srcid=0327VOg7SqZ0uCzzWGbuLIeC&pass_ticket=T5Eg3z3g4gegrRw8o9ppg8EGQHVx0p9nktLFYFVZ1A5MUS%2BDyASZSJuGl2tzoDv5#rd)以及**自己做的PPT**

作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!