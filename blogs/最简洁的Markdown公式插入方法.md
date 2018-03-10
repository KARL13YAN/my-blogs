# CSDN最简洁的Markdow公式编辑及插入方法

Markdown是一种可以使用普通文本编辑器编写的标记语言，通过简单的标记语法，它可以使普通文本内容具有一定的格式。CSDN作为一个技术论坛，在撰写文章的时候难免需要插入数学公式。但是一个字一个字的写公式真的非常让人头疼！！下面介绍一个**简洁的公式插入方法**，保证让你爱不释手！

## 1. Markdown公式基础

首先，你必然要会编辑最简单的公式，譬如$k^2, x_i$等等，推荐一个良心网站：作业部落，[Markdown 简明语法手册][1]以及[Markdown 公式指导手册][2]，点击跳转。

## 2. 正式教学 :smirk:

我们需要借助这个网站：[Latex][3]，界面如下：

![codecogs网站界面][4]

**1**区可以选择需要编辑的公式，**2**区可以编辑该公式，公式的预览会出现在**4**区，点击**5**区可以复制公式，在Markdown编辑器中粘贴到两个美元符号的中间，即可展示该公示。此外，**6**区还可以直接下载该公式的图片，你可以选择是gif/png/jpg/svg等多种格式。

公式还可以直接以图片的格式展示在文章中，你所要做的，仅仅是在**7**区中选择**URL**,然后复制下面文本框的内容

![公式](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}x_i^2)

上面这个公式的Markdown代码是：

    ![公式](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}x_i^2)

对比一下不以图片形式展示的公式：$\sum_{i=1}^{n}x_i^2$

我觉得图片形式更OK！

---
作者邮箱： mr.yxj@foxmail.com

转载请告知作者，感谢!

[1]:https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown
[2]:https://www.zybuluo.com/codeep/note/163962
[3]:http://www.codecogs.com/latex/eqneditor.php
[4]:https://github.com/KARL13YAN/learning/raw/master/pic/2.png