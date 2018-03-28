# Python机器学习读书笔记（十）机器学习模型的评价指标

> 说明：
>
> 1. 关于本书: [《Python机器学习》](https://book.douban.com/subject/27000110/)
> 2. 本笔记侧重代码调用，只描述了一些简单概念，本书的公式推导不在这里展示

---

- 前面的章节中，都是用准确性对模型进行评估。通常情况下，准确性确实是一个有效量化模型的指标

- 下面介绍其它几个可以用来衡量模型相关性能的指标，比如**准确率 precision**、**召回率 recall**和**F1分数 F1-score**

## 1. 混淆矩阵 confusion matrix

- 混淆矩阵可以展示学习算法的性能，是一个简单的方阵，用于展示一个分类器的预测结果

1. **真正 true positive**：预测为1，实际为1
2. **真负 true negative**：预测为0，实际为0
3. **假正 false positive**：预测为1，实际为0
4. **假负 false negative**：预测为0，实际为1

![confusion matrix](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_08.png)

- sklearn提供了`confusion_matrix`函数来得到混淆矩阵

```py
from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


[[71  1]
 [ 2 40]]
```

- 可以使用`matplotlib`中的`mathshow`函数将它们表示出来

```py
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')

plt.tight_layout()
# plt.savefig('./figures/confusion_matrix.png', dpi=300)
plt.show()
```

![mat_c_m](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_09.png)

- 这些信息是计算下一节中误差度量的基础

## 优化分类模型的准确率和召回率

1. **预测误差（error，ERR）**，可以理解为 *预测错误样本的数量* 与 *所有被预测样本数量* 的比值

    - $$ERR=\frac{FP+FN}{FP+FN+TP+TN}$$

2. **准确率（accuracy，ACC）**，*预测正确样本的数量* 与 *所有被预测样本数量* 的比值

    - $$ERR=\frac{TP+TN}{FP+FN+TP+TN}=1-ERR$$

3. **假正率 FPR**对类别不均衡的分类问题来说比较有用

    - $$FPR=\frac{FP}{N}=\frac{FP}{FP+TN}$$

4. **真正率 TPR**对类别不均衡的分类问题来说比较有用。

    - $$TPR=\frac{TP}{P}=\frac{TP}{FN+TP}$$

5. **准确率 precision PRE**

    - $$PRE=\frac{TP}{TP+FP}=\frac{TP}{FN+TP}$$

6. **召回率 recall REC**

    - $$REC=TPR=\frac{TP}{P}=\frac{TP}{FN+TP}$$

7. **F1-score**准确率和召回率的组合

    - $$F1=2*\frac{PRE*REC}{PRE+REC}$$

- sklearn已经实现以上评分指标，sklearn将正类标识为1

```py
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
```

## ROC曲线 receiver operator characteristic

- 基于模型假正率和真正率等性能指标**进行分类模型选择**的有用工具

- 假正率和真正率可以通过移动分类器的分类阈值来计算
- ROC的对角线可以理解为随即猜测，**如果分类器性能曲线在对角线下，那么其性能比随机猜测还要差**
- 对于完美的分类器来说，**真正率为1，假正率为0**，此时的ROC曲线是横轴和纵轴组成的折线。
- 基于ROC曲线，可以计算ROC线下区域（area under the curve，AUC），用来刻画分类模型性能

```py
from sklearn.metrics import roc_curve, auc
from scipy import interp

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(penalty='l2',
                                               random_state=0,
                                               C=100.0))])

X_train2 = X_train[:, [4, 14]]


if Version(sklearn_version) < '0.18':
    cv = StratifiedKFold(y_train,
                         n_folds=3,
                         random_state=1)

else:
    cv = list(StratifiedKFold(n_splits=3,
                              random_state=1).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             lw=1,
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2,
         linestyle=':',
         color='black',
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
# plt.savefig('./figures/roc.png', dpi=300)
plt.show()
```

![roc](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/images/06_10.png)

- 仅计算ROC AUC

```py
pipe_lr = pipe_lr.fit(X_train2, y_train)
y_labels = pipe_lr.predict(X_test[:, [4, 14]])
y_probas = pipe_lr.predict_proba(X_test[:, [4, 14]])[:, 1]
# note that we use probabilities for roc_auc
# the `[:, 1]` selects the positive class label only

rom sklearn.metrics import roc_auc_score, accuracy_score
print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_probas))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_labels))

ROC AUC: 0.752
Accuracy: 0.711
```

## 多类别分类的评价标准

- sklearn内部实现了macro及micro均值方法，可以通过（One vs All，OvA）的方式将评价标准扩展到多类别分类问题

- micro均值是通过系统的真正、真负、假正、假负来计算；等同看待每个实例或每次预测
- 例如，k类分类系统的准确率评分的micro均值可以按如下公式计算

$$PRE_{micro}=\frac{TP_1+...+TP_k}{TP_1+...+TP_k+FP_1+...+FP_k}$$

- macro均值仅计算不同系统的平均分值；等同看待每个类别

$$PRE_{macro}=\frac{PRE_1+...+PRE_k}{k}$$

```py
pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')
```