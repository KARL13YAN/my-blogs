# Python机器学习读书笔记（十二）网格搜索&嵌套交叉验证

> 说明：
>
> 1. 关于本书: [《Python机器学习》](https://book.douban.com/subject/27000110/)
> 2. 本笔记侧重代码调用，只描述了一些简单概念，本书的公式推导不在这里展示

---

## **网格搜索**

- 机器学习中的两类参数：
  1. 通过训练得到的参数（如Logistic回归中的回归参数）
  2. 算法中需要单独进行优化的参数，即**超参**（如Logistic回归中的正则化系数）

- 在[前文](https://blog.csdn.net/weixin_40604987/article/details/79676066)中介绍了使用验证曲线调优超参提高模型性能
- 本文介绍了一种功能强大的超参优化技巧：**网格搜索 Grid Search**，它通过寻找最优的超参值的组合进一步提高模型性能

### **原理**

- 通过对我们人为指定的不同超参列表进行暴力穷举搜索，并计算评估每个组合对模型性能的影响，以获得最优的参数组合

### **sklearn调用**

```py
from sklearn.svm import SVC
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
                 {'clf__C': param_range,
                  'clf__gamma': param_range,
                  'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

0.978021978022
{'clf__kernel': 'linear', 'clf__C': 0.1}

```

- 本文调用了SVM模型
  1. 对线性SVM来说，只需要调优正则化参数`C`
  2. 对RBF核SVM来说，需要同事调优`C`和`gamma`参数

- 初始化了一个`GridSearchCV`对象，用于对SVM流水线的训练与调优
- `GridSearchCV`的`param_grid`参数以字典的方式定义为待调优的参数
- 在训练集数据上完成网格搜索后，可以通过`best_score_`属性得到最优模型的性能评分
- 在本例中，线性SVM模型在`clf__C=0.1`时，可得到最优k折交叉验证准确率97.8%

- 随后，用独立测试集对最优模型进行性能评估

```py
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))


Test accuracy: 0.965
```

- **TIPS**:虽然网格搜索是寻找最优参数组合的强大方法，但这种暴力搜索的办法需要的计算成本也是巨大的。借助sklearn中的`RandomizedSearchCV`类，我们可以以特定的代价从抽样分布中抽取出随机的参数组合。

## **通过嵌套交叉验证选择算法**

- 如果需要在不同机器学习算法之间做选择，则可以使用**嵌套交叉验证**

### **原理 **

- 在嵌套交叉验证的**外围循环**中，将数据划分为训练块及测试块
- 在**内部循环**中，则基于这些训练块，使用k折交叉验证
- 完成模型选择后，使用测试块进行模型性能的评估
- 如下图所示，是一种5*2交叉验证

![yanzheng](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_07.png)

### **sklearn调用 **

```py
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

# Note: Optionally, you could use cv=2
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.

scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

CV accuracy: 0.965 +/- 0.025
```

- 也可以使用该方法比较模型。如比较SVM模型和决策树模型

```py
from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


CV accuracy: 0.921 +/- 0.029
```