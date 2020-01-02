# Daily_Paper
This project is for reading a paper everyday

### 2020.01.02
[diffGrad: An Optimization Method for Convolutional Neural Networks](https://arxiv.org/abs/1909.11015)

文章提出SGD的问题在于迭代过程中相等的步长，不会根据梯度表现自适应。AdaGrad、AdaDelta、RMSProp、Adam都在提高梯度下降的方法，然而这些方法依赖于梯度平方的指数移动平均的平方跟，并没有利用到梯度的局部变化。本文提出了一种新的diffGrad，梯度下降的步长会自适应变化，并给出了收敛行证明。并公开了代码在 https://github.com/shivram1987/diffGrad

diffGrad将当前和过去迭代的梯度差异（即短期梯度变化信息）与Adam优化技术结合在一起来控制优化过程的学习率。diffGrad在梯度变化大的时候得到一个更高的学习率，在梯度变化小的地方得到一个低的学习率。为了避免陷入局部最优或者鞍点，由惯性冲量moment来控制。

作者是几个IEEE，文中有其收敛性证明，先码起来，周末再研究一下，他人的学习笔记可以先参考：https://youyou-tech.com/2019/12/28/%E8%AE%A4%E8%AF%86DiffGrad%EF%BC%9A%E6%96%B0%E5%9E%8B%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%BC%98%E5%8C%96%E5%99%A8/

### 2020.01.03
[PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)

### 2020.01.04
[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
Swish

