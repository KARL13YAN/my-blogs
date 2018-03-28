# Python机器学习读书笔记（十三）——正则化

>[前文](https://blog.csdn.net/weixin_40604987/article/details/79676066)提到了方差与偏差、使用验证曲线发现欠拟合和过拟合的问题，并提出了[网格搜索](https://blog.csdn.net/weixin_40604987/article/details/79691752)、[交叉验证](https://blog.csdn.net/weixin_40604987/article/details/79644396)等方法解决问题。
>事实上，我们还可以通过**正则化**来解决这个问题。

## 引入

![拟合](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_06.png)

- 如上图所示
  1. 左边是欠拟合（高偏差）的情况，意味着模型过于简单，模型应用于未知数据时可能效果不佳
  2. 中间是拟合良好的情况
  3. 右边是过拟合（高方差）的情况，意味着模型过于复杂、参数过多

- 解决上述问题的一个方案就是**正则化**。正则化可以调整模型的复杂度，是解决共线性问题的一个很有用的办法，可以过滤掉数据中的噪声，最终防止过拟合

- **原理**：引入额外的信息对极端参数权重做出惩罚

- 常用的正则化方法包括：L1正则化、L2正则化

- 权重更新函数：$w'=w-\eta\Delta C(w)$

## L1正则化

- 在原始的代价函数$C$后面加上一个L1正则化项，即所有权重w的绝对值的和乘以$λ/n$

$$C=C_0+\frac{\lambda}{n}\sum_m|w|$$

- 计算导数

$$\frac{dC}{dw}=\frac{dC_0}{dw}+\frac{\lambda}{n}sgn(w)$$

- 权重w的更新公式

$$w\rightarrow w'=w-\frac{\eta\lambda}{n}sgn(w)-\eta\frac{dC_0}{dw}$$

- 比原始的更新规则多出了$\frac{\eta\lambda}{n}sgn(w)$这一项
  1. 当w为正时，更新后的w变小
  2. 当w为负时，更新后的w变大

- 它的效果就是让w往0靠，使网络中的权重尽可能为0，也就相当于减小了网络复杂度，防止过拟合

- w等于0时，$|w|$是不可导的，所以我们只能按照原始的未经正则化的方法去更新w，这就相当于去掉$\frac{\eta\lambda}{n}sgn(w)$这一项，所以我们可以规定$sgn(0)=0$，这样就把$w=0$的情况也统一进来了。

### 使用sklearn进行L1正则化

- 下面的例子在波士顿房价数据上运行了Lasso，其中参数alpha是通过grid search进行优化的

- 可以看到，很多特征的系数都是0。如果继续增加alpha的值，得到的模型就会越来越稀疏，即越来越多的特征系数会变成0

```py
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
Y = boston["target"]
names = boston["feature_names"]

lasso = Lasso(alpha=.3)
lasso.fit(X, Y)

print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)

# Lasso model: -3.707 * LSTAT + 2.992 * RM + -1.757 * PTRATIO + -1.081 * DIS + -0.7 * NOX + 0.631 * B + 0.54 * CHAS + -0.236 * CRIM + 0.081 * ZN + -0.0 * INDUS + -0.0 * AGE + 0.0 * RAD + -0.0 * TAX
```

## L2正则化

- L2正则化就是在代价函数后面再加上一个正则化项：

$$C=C_0+\frac{\lambda}{2n}\sum_m|w^2|$$

- 所有参数$w$的平方的和，除以训练集的样本大小$n$,$λ$就是正则项系数，权衡正则项与$C_0$项的比重。

- 为什么会有系数$1/2$？仅仅是为了后面求导的方便而已

- 求导

$$\frac{dC}{dw}=\frac{dC_0}{dw}+\frac{\lambda}{n}w$$

- 权重更新

$$w\rightarrow w'=w-\frac{\eta\lambda}{n}w-\eta\frac{dC_0}{dw}=(1-\frac{\eta\lambda}{n})w-\eta\frac{dC_0}{dw}$$

- 在不使用L2正则化时，求导结果中$w$前系数为1，现在$w$前面系数为$1-\frac{\eta\lambda}{n}$ ，因为$η,λ,n$都是正的，所以$1-\frac{\eta\lambda}{n}$小于1，它的效果是减小$w$，这也就是权重衰减（weight decay）的由来。当然考虑到后面的导数项，$w$ 最终的值可能增大也可能减小。

- 更小的权值w，从某种意义上说，表示网络的复杂度更低，对数据的拟合刚刚好（这个法则也叫做奥卡姆剃刀），而在实际应用中，也验证了这一点，L2正则化的效果往往好于未经正则化的效果

### 使用sklearn进行L2正则化

- 下面是经过L2正则化的Logistic回归
- **请注意**，L2正则化还可以这样写,也就是把正则化参数提到前面去

$$J(w)=C*J+\frac{1}{2}\sum_m|w^2|$$

```py
weights, params = [], []
for c in np.arange(-5., 5.):
    lr = LogisticRegression(C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()
```

![正则化](https://github.com/rasbt/python-machine-learning-book/raw/master/code/ch03/images/03_23.png)

- 可以看到，减小C（也就是增加正则化强度），可以导致权重系数的衰减

---
**参考**：

1. [正则化方法](http://mp.weixin.qq.com/s?__biz=MzA4OTg5NzY3NA==&mid=2649345393&idx=1&sn=f41d44f7a492a1534b81e86815633f91&chksm=880e82d4bf790bc25c207084c2a8761df18c293ffac1603ce7895af824e1235f994a7560fc9e&mpshare=1&scene=1&srcid=03266dPucYkS1OTVtPfJENav#rd)
2. [机器学习](https://book.douban.com/subject/26708119/)

---

作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!