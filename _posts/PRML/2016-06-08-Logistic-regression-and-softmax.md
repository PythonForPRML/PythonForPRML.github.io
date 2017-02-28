---
layout: article
title:  "Logistic regression and softmax"
categories: prml
time: 2016-6-18
---

# 一、逻辑回归

逻辑回归主要是处理二分类问题，假设我们的score function采用线性函数表示，即 $s = X \cdot W$,其中X为$N \times D$(N个样本，每个样本D维)，而W为$D \times 1$

那么我们所求得的 $s = \{s_1,s_2,...,s_N \}^T$, 其中 $s_j$表示第j个输入的score。

我们使用sigmoid函数对输入的score function 进行处理，那么所有的score都处于0到1之间
而我们只有两个分类，$C_1$,$C_2$

令   
$$p(C_1 \vert x) = \sigma (x\cdot W)$$
那么
$$p(C_2 \vert x) = 1- p(C_1 \vert x)$$




## sigmoid 函数的性质

$\sigma(x) = \frac{1}{1 + e^{-x}}$

$\sigma(x)^{'} = (1- \sigma(x))\sigma(x) $

## 逻辑回归的损失函数及其对W的偏导数


对于一个数据集 $\{X_N,t_N\}$,其中 $t_n \in {0,1},n = 1,2...N$

样本分布为 $X = [ X_1^T, X_2^T,...,X_N^T]^T$, $X_i$表示第i个样本，即X这个矩阵中的第一行。

其似然函数可以写作：

$$p(\mathbf{t} \vert \mathbf{w}) = \prod_{n=1}^{N}y_n^{t_n}\{1-y_n\}^{1-t_n}$$

其中  $\mathbf{t} = \{t_1,....t_N\}^T$ 且  $y_n = p(C_1 \vert x_n)$

我们将loss function定为似然函数的负对数，这会导出 cross-entropy loss function:

$$E(\mathbf{w}) = -lnp(\mathbf{t} \vert \mathbf{w}) = - \sum_{n=1}^{N} \{ t_n ln y_n + (1-t_n) ln (1-y_n)\}$$

我们使用链式法则对其求对W的偏导数。

我们定义 $y_n = \sigma (a_n)$,$a_n = W^Tx_n$

$$\frac{\partial E}{\partial y_n} = \frac{1-t_n}{1-y_n} - \frac{t_n}{y_n} = \frac{y_n -t_n}{y_n ( 1-y_n )}
$$

$$\frac{\partial y_n}{\partial a_n} =\frac{\partial \sigma (a_n)}{\partial a_n} = y_n(1-y_n)$$

且
$$\frac{\partial a_n}{\partial W} = X_n^T$$

那么 $$\nabla E(W) = \frac{1}{N}\sum_{n=1}^{N}(y_n - t_n)X_n^T$$

我们可以看到  

> 损失函数对参数W的偏导是输入的线性组合，每个输入样本$X_n^T$对参数W偏导的贡献系数为分类错误 $y_n -t_n$

> 我们将看到很多损失函数的偏导都有着上述形式


> 工程实现时，我们可以如下计算

$$\nabla E(W) = \frac{1}{N}[X_1^T,X_2^T,...X_N^T]\begin{bmatrix}y_1-t_1 \\ y_2-t_2 \\ \vdots \\ y_n-t_n \end{bmatrix}=\frac{1}{N} X^T(\mathbf{y}-\mathbf{t}) = \frac{1}{N}X^T(\sigma{(X \cdot W)}-\mathbf{t}) $$

其实我们可以将上式表示为：

$$\nabla E(W) =  \frac{1}{N}X^T ( \sigma{(X \cdot W)}- Indicator(\mathbf{t} = 1) ) $$



# 二、softmax

在深度学习中，通常用softmax来作为最后一层并计算损失函数。


## softmax 函数

对于一个向量z，经过soft函数的输出为：

$f(z_i) = \frac{ e^{z_i}} { \sum_{j} { e^{z_j} } } $


假设我们使用softmax来进行C个类别的分类问题，输入为X,参数为W，其中X为$N \times D$(N个样本，每个样本D维)，而W为$D \times C$

令 $y = X \cdot W$, 那么y的大小为 $N \times C$

softmax的输出为

$f(y_{ij}) = \frac{ e^{y_{ij}}} { \sum_{j} { e^{y_{ij}} } }$ 

$y_{ij}$表示第i次输入时第j个类别的取值，因此$f(y_{ij})$表示的是第i次输入时第j个类别的概率。

我们采用cross-entropy来计算softmax的loss函数。

## cross-entropy

交叉熵主要是用来衡量真实分布p和估计分布q之间的一致程度的。

$$H(p,q) = - \sum_{x} p(x)logq(x)$$

对于第i次输入，其所属类别为$t_i$,那么在第i次输入时其真实分布为 $p = [0,0,...1,...,0]$(仅仅在$t_i$处值为1)。而softmax模型对第i次的估计概率为$f(y_{ij})$，那么对于第i次输入其交叉熵(损失函数)为：

$$L_i = -\sum_{j} p_j log(f(y_{ij})) = -logf(y_{it_i})= - log( \frac{ e^{y_{it_i}}} { \sum_{j} { e^{y_{ij}} } } )$$

或者将上式展开得到：

$$L_i = -y_{it_i} + log(\sum_{j}{e^{y_{ij}}})$$
其中$f(y_{it_i})$表示第i输入时正确分类的取值


为了求出loss对参数W的偏导，我们可以先求出loss对$y_{ij}$的取值

$$\frac{\partial L_i}{\partial y_{ij}} = \frac{ e^{ y_{ij} } } {\sum_{j}{e^{y_{ij}}}} - Indicator(j = t_i) = f(y_{ij}) - Indicator(j = t_i)$$

而$y_{ij}$对第j个类的参数$W_j$的偏导为 

$$\frac{\partial y_{ij}}{\partial W_j} = X_i^{T}$$

在第i次输入时loss对第j个类的参数W的偏导为：

$$\frac{\partial L_i}{\partial W_j} = \frac{\partial L_i}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial W_j} = X_i^{T} \cdot \left( f(y_{ij}) - Indicator(j = t_i) \right) $$

那么我们可以得到在一个batch中输入为X的时候loss对W的偏导：

$$ \frac{\partial L}{\partial W} = \frac{1}{N} X^{T}  \left( f(X \cdot W) - Indicator( j = \mathbf{t} ) \right) $$

# 逻辑回归和softmax的联系

### 1. 两者的loss函数都可以用cross-entropy得到

我们在前文中对logistic regression的loss函数的推导是由对似然函数取对数得到的，但实际上我们也可以直接来计算真实分布和估计分布之间的交叉熵来得到：

我们用sigmoid函数来表示模型对样本的分类为1和0的概率:

$$q(t|x) =   \begin{cases}
    \sigma(X \cdot W)       & \quad \text{if } t = 1\\
    1- \sigma(X \cdot W)  & \quad \text{if } t = 0\\
  \end{cases}$$
  
那么在第i次输入的情况下，如果真实分类为1，错误分类为0,此时 p(t=1) = 1， p(t=0) = 0，其损失函数为：

$$L_i =- \sum_{x} p(x)logq(x) = - p(t=1)log \left(q(t=1|x)\right) - p(t=0)log \left(q(t=0|x)\right) = -log( \sigma(X \cdot W))$$


那么在第i次输入的情况下，如果真实分类为0，错误分类为1,此时 p(t=1) = 0， p(t=0) = 1，其损失函数为：

$$L_i =- \sum_{x} p(x)logq(x) = - p(t=1)log \left(q(t=1|x)\right) - p(t=0)log \left(q(t=0|x)\right) = -log( 1- \sigma(X \cdot W))$$

为了把上面两种情况结合起来，我们可以将$L_i$表示成：

$$L_i = -t_n \sigma(X \cdot W) - (1-t_n)( 1- \sigma(X \cdot W))$$

### 2. 两者对参数W的偏导都是每次输入的线性组合

logistic regression对参数W（只有一个类）的偏导为:

$$\nabla E(W) =  \frac{1}{N}X^T ( \sigma{(X \cdot W)}- Indicator(\mathbf{t} = 1) ) $$

softmax 对参数W的偏导为：
$$ \frac{\partial L}{\partial W} = \frac{1}{N} X^{T}  \left( f(X \cdot W) - Indicator( j = \mathbf{t} ) \right) $$

我们可以看出：

> 损失函数对参数W的偏导是多个输入的线性组合

> 每个输入样本$X_n$对参数W偏导的贡献系数为 $score(X_n) - Indicator(当前预测类和真实类一致)$

因为$score(X_n)$小于0，所以负例对参数W偏导的贡献系数为正，而正例d的系数为负，

不过因为 $W^{new} = W^{old} - \alpha dW$,所以最终情况是，

如果是正例，在原W上加上当前输入，如果是负例，从原W上减去当前输入，当然一定的系数是需要被考虑的。




```python

```
