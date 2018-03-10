## Python获取元素出现次数
----
### 1. 前言

在使用Python的时候，通常会出现如下场景：

    array =  [1, 2, 3, 3, 2, 1, 0, 2]

    获取array中元素的出现次数

比如，上述列表中：0出现了1次，1出现了2次，2出现了3次，3出现了2次。

本文阐述了Python获取元素出现次数的几种方法。[点击获取完整代码](https://github.com/KARL13YAN/learning/blob/master/get_frequency.py)。

### 2. 方法
获取元素出现次数的方法较多，这里我提出如下5个方法，谨供参考。下面的代码，传入的参数均为 `array =  [1, 2, 3, 3, 2, 1, 0, 2]`

#### 2.1 Counter方法
该方法可以迅速获取list中元素出现的次数，可以参考[官方文档](https://docs.python.org/3/library/collections.html)
```
from collections import Counter
def counter(arr):
    return Counter(arr).most_common(2) # 返回出现频率最高的两个数

# 结果：[(2, 3), (1, 2)]
```
#### 2.2 list中的count,获取每个元素的出现次数
```
def single_list(arr, target):
    return arr.count(target)

# target=2，结果：3
```

#### 2.3 list中的count,获取所有元素的出现次数
返回一个dict
```
def all_list(arr):
    result = {}
    for i in set(arr):
        result[i] = arr.count(i)
    return result

# 结果：{0: 1, 1: 2, 2: 3, 3: 2}
```
#### 2.4 Numpy花式索引，获取每个元素的出现次数
```
def single_np(arr, target):
    arr = np.array(arr)
    mask = (arr == target)
    arr_new = arr[mask]
    return arr_new.size

# target=2，结果：3
```
#### 2.5 Numpy花式索引，获取所有元素的出现次数
返回一个dict
```
def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

# 结果：{0: 1, 1: 2, 2: 3, 3: 2}
```
### 3. 总结
以上就是我总结的几种Python获取元素出现个数的方法。

值得一提的是，我所用的list所有元素都是整数

`array =  [1, 2, 3, 3, 2, 1, 0, 2]`

如果list中包含其它类型的元素，比如

`array =  [1, 2, 3, 3, 2, 1, 'a', 'bc', 0.1]`

这种情况下需要获取 `a`或`1` 的出现次数时，2.4中函数的调用形式应当为：`target='a'` / `target='1'`

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!