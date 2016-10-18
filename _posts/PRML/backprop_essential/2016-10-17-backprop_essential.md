---
layout: article
title:  "Mathmatics behind back propagation"
categories: prml
time: 2016-10-17
---

# 反向传播求偏导的数学本质

本文拟解决一直困扰许多人的反向传播过程中对矩阵的求导，在这篇论文中，我将详细解释每一步操作背后的数学原理，在学会这篇文章中的内容之后，你可以轻松地写出任何layer的反向传播。

## 一、纠正错误
在神经网络中，只有实数对矩阵的偏导，而不存在矩阵对矩阵的偏导。对于反向传播的所有的求导，我们都要转化成实数对矩阵的偏导。

## 二、一些基本的矩阵运算和求导法则

### 1.矩阵乘法

矩阵乘法有很多不同的表示方式，此处为了便于本文的解释说明，采取列的线性组合这一角度来进行介绍。

#### 1. 矩阵右乘列向量——对矩阵的每一列做线性组合

假设A是一个 $$n \times m$$的矩阵，x是一个列向量，我们将A表示成列的形式，那么：

$$
y = Ax = \begin{bmatrix}
            | & | &  &| \\
            a_1 & a_2 & ... & a_m \\
            | & | &  & |  \\
            \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ ... \\ x_m \\ \end{bmatrix} = \begin{bmatrix} \\ a_1 \\ \\ \end{bmatrix}x_1 + \begin{bmatrix} \\ a_2 \\ \\ \end{bmatrix}x_2 + ... + \begin{bmatrix} \\  a_m \\ \\ \end{bmatrix}x_m
$$

注意此处y是A中的每一列的线性组合，其相应系数为x中的对应元素， y的大小为n*1


#### 2. 矩阵相乘

我们可以将右矩阵的每一列与左矩阵相乘， A为$$n \times k$$ ，B为$$k \times m$$, 那么

$$
C = AB = A\begin{bmatrix}
            | & | &  &| \\
            b_1 & b_2 & ... & b_m \\
            | & | &  & |  \\
        \end{bmatrix} = 
        \begin{bmatrix}
            | & | &  &| \\
            Ab_1 & Ab_2 & ... & Ab_m \\
            | & | &  & |  \\
        \end{bmatrix}
$$



### 2.实数对矩阵的偏导

假设$$y \in R$$,  $$X \in R^{m \times n}$$,那么 y对X的求导$$\frac{d y}{dX}$$:

$$\begin{bmatrix}
    \frac{d y}{d X_{11}} & ... & \frac{d y}{d X_{1n}} \\
    \vdots & \vdots & \vdots \\
    \frac{d y}{d X_{m1}} & ... & \frac{d y}{d X_{mn}}
\end{bmatrix}
$$

我们可以看到，实数对矩阵求导的结果和矩阵的大小一样

# 三、 神经网络review

神经网络中的基本单元是一个神经元，每个神经元接收上一层多个神经元的信号，不过这些信号都有一个权重。我们可以简单地认为这个权重大于0的时候前一个神经元的输出被传播到当前神经元中，而当权重小于0的时候前一个神经元的输出就被截断了。

<img src = "/images/PRML/backprop_essential/neuron_inputs.png">

而我们在神经网络中接触较多的却是layer这一概念，将多个相同类型的神经元组合成一个layer。 但实际上，某个layer上的每个神经元都有其自身的一个权重，而且互相没有任何关系(递归网络的神经元之间可能有联系，但也适用本文所说的基本思想)，表示成layer只是为了方便整个网络的结构表示。


所以在我们学习神经网络的时候，应该去关注每个神经元的操作，这样才能对各个数学操作背后的本质有所了解。

为了说明反向传播中求导的问题，我们先以全连接层来说明。

<img src = "/images/PRML/backprop_essential/fully_connected.png">

假设输入层有D个神经元，全连接层有C个神经元。我们可以将输入表示成$$1 \times D$$的向量，而全连接层的每一个神经元都有D个权重，我们将其表示成$$D \times C$$的矩阵，其第i列表示第i个神经元的权重。

那么我们可以得到：

$$
\begin{bmatrix} x_1 & x_2 & ... & x_D \end{bmatrix} \begin{bmatrix} W_{11} & W_{21} & ... & W_{C1} \\  W_{12} & W_{22} & ... & W_{C2}  \\ \vdots & \vdots  & \ddots & \vdots \\  W_{1D} & W_{2D} &  ... & W_{CD} \end{bmatrix} = \begin{bmatrix} y_1 & y_2 & \ddots & y_C \end{bmatrix}
$$

其中$$W_{ij}$$表示第i个神经元的第j个权重, 我们将第j个神经元的权重表示成$$W_j = \begin{bmatrix}W_{j1} & W_{j2} & ... & W_{jD}\end{bmatrix}^T$$

用矩阵来表示一个layer上所有的神经元的计算结果只是为了计算过程的简便，实际上第j个神经元的值$$y_j$$只和其本身的权重有关：

$$
\begin{bmatrix} x_1 & x_2 & ... & x_D \end{bmatrix} \begin{bmatrix}  W_{j1} \\ W_{j2}  \\ \vdots \\   W_{jD} \end{bmatrix} = \begin{bmatrix}  y_j \end{bmatrix}
$$

那么如果我们要求$$y_j$$对$$W_j$$的偏导，根据上面我们所说的实数对矩阵的求导法则：

$$
\frac{d y_j}{d W_j} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_D \end{bmatrix}
$$

## 1. 反向传播的过程

在神经网络中，每一层的输出都会被前向传播到后一层，在最后一层的时候，为了刻画这个模型的好坏程度，我们会定义一个损失函数(loss function)。损失函数的值用实数表示，然后使用反向传播算法来将loss value对每一层的偏导传播到前一层。

<img src = "/images/PRML/backprop_essential/backProp.png">

当我们使用反向传播算法来优化神经网络的时候，我们需要在每一层上将该层的权重W减去该层输出对W的偏导dW。

还是以全连接层为例，假设从后一层反向传播回来的，对第j个神经元的偏导数用$$\frac{d L}{d y_j}$$表示，那么第j个神经元的权重更新为(learning rate 为 $$\alpha$$)：

$$
W_j^{new} = W_j^{old} - \alpha \cdot  dW_j = W_j^{old} - \alpha \frac{d L}{d y_j} \frac{d y_j}{d W_j}
$$

我们的目标是求得$\frac{d L}{d y_j} \frac{d y_j}{d W_j}$，因为$\frac{d L}{d y_j}$是一个实数，我们将其简记为$$y_j^{'}$$，利用我们之前的推到过程:

$$
\frac{d L}{d y_j} \frac{d y_j}{d W_j} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_D \end{bmatrix}y_j^{'} = \begin{bmatrix} x_1 \cdot y_j^{'} \\ x_2 \cdot y_j^{'} \\ \vdots \\ x_D \cdot y_j^{'} \end{bmatrix}
$$

这是对单个神经元的参数求导表示，为了简单第表示整个layer上所有神经元的参数更新，我们可以如前文所述将整个神经元表示成一个矩阵：

$$
\begin{bmatrix} W_{11} & W_{21} & ... & W_{C1} \\  W_{12} & W_{22} & ... & W_{C2}  \\ \vdots & \vdots  & \ddots & \vdots \\  W_{1D} & W_{2D} &  ... & W_{CD} \end{bmatrix}  = \begin{bmatrix} W_1 & W_2 & \cdots W_j \end{bmatrix}
$$

其中$$W_{ij}$$表示第i个神经元的第j个权重, 我们将第j个神经元的权重表示成$$W_j = \begin{bmatrix}W_{j1} & W_{j2} & ... & W_{jD}\end{bmatrix}^T$$

那么我们可以将整个layer的参数同时更新

$$
W^{new} = W^{old} - \alpha \cdot  dW = W^{old} - \alpha \frac{d L}{d y} \frac{d y}{d W}
$$

此时我们可以将 $$\frac{d L}{d y} \frac{d y}{d W}$$表示成：

$$
\begin{bmatrix} x_1 \cdot y_1^{'} & x_1 \cdot y_2^{'} & ... & x_1 \cdot y_C^{'} \\  x_2 \cdot y_1^{'} & x_2 \cdot y_2^{'} & ... & x_2 \cdot y_C^{'}  \\ \vdots & \vdots  & \ddots & \vdots \\  x_D \cdot y_1^{'} & x_D \cdot y_2^{'} &  ... & x_D \cdot y_C^{'} \end{bmatrix}  = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_D \end{bmatrix} \begin{bmatrix} y_1^{'} & y_2^{'}  & \dots &  y_C^{'} \end{bmatrix} = X_1^T \begin{bmatrix} y_1^{'} & y_2^{'}  & \dots &  y_C^{'} \end{bmatrix}
$$

我们可以看出同一层上C个神经元的参数W的更新量都与该层的输入成正比， 不同之处仅在于系数不同，第j个神经元的更新系数为$$y_j{'}$$。而$$y_j{'}$$表示的是后一层对第j个神经元输出值的偏导

## 2. 多个输入时的反向传播

目前为止，我们已经推导出用矩阵来表示整个全连接层上所有神经元在反向传播中的参数更新。但是这仅仅针对一次输入而言，在神经网络中，我们往往会在一个batch中同时传入多个输入。我们将第k次输入表示为$$X_k = \begin{bmatrix} X_{k1} & X_{k2} & \dots & X_{kD} \end{bmatrix}$$.

我们可以在每一次输入时进行一次反向传播：

$$
\text{for i from 1 to N:} \\
        W^{new} = W^{old} - \alpha \cdot  dW = W^{old} - \alpha \frac{d L}{d y} \frac{d y}{d W}      
$$


但是在神经网络中，我们更倾向于用矩阵而非循环，那么如何才能表示多次输入时的layer的参数更新呢？

首先我们来看对于单个神经元的多次输入， 首先分析最简单的情况，只有两次输入，第一次输入为$$X_1$$,第二次输入为$$X_2$$，那么对于第j个神经元我们有：


第一次输入$$X_1$$时

$$
\frac{d L}{d y_{1j}} \frac{d y_{1j}}{d W} = \begin{bmatrix} X_{11} \\ X_{12} \\ \vdots \\ X_{1D} \end{bmatrix} \cdot y_{1j}^{'}
$$

其中 $$y_{1j}$$表示第1次输入时第j个神经元从后一层反向传播回来的偏导。

第二次输入$$X_2$$时

$$
\frac{d L}{d y_{2j}} \frac{d y_{2j}}{d W} = \begin{bmatrix} X_{21} \\ X_{22} \\ \vdots \\ X_{2D} \end{bmatrix} \cdot y_{2j}^{'}
$$

其中 $$y_{2j}$$表示第2次输入时第j个神经元从后一层反向传播回来的偏导。

因为每次输入都会进行一次反向传播，而每次反向传播都会将第j个神经元的当前参数W减去dW，那么我们可以将两次输入的效果叠加起来：


$$
\frac{d L}{d y_{j}} \frac{d y_{j}}{d W} = \sum_{k = 1}^{2} \frac{d L}{d y_{kj}} \frac{d y_{kj}}{d W} = \begin{bmatrix} X_{11} \\ X_{12} \\ \vdots \\ X_{1D} \end{bmatrix} \cdot y_{1j}^{'} + \begin{bmatrix} X_{21} \\ X_{22} \\ \vdots \\ X_{2D} \end{bmatrix} \cdot y_{2j}^{'} = \begin{bmatrix} X_{11} & X_{21} \\ X_{12} & X_{22} \\ \vdots & \vdots \\ X_{1D} & X_{2D}  \end{bmatrix} \begin{bmatrix} y_{1j}^{'} \\ y_{2j}^{'} \end{bmatrix} = \begin{bmatrix} X_1^T & X_2^T  \end{bmatrix} \begin{bmatrix} y_{1j}^{'} \\ y_{2j}^{'} \end{bmatrix}
$$


我们可以看出第j个神经元的参数在反向传播时的更新量只与本次输入有关，其系数为该神经元从后一层反向传播回来的偏导。 多次输入的效果可以线性叠加。



至此，我们已经探讨过：

第j个神经元在多次输入时的参数更新量为：

$$
\begin{bmatrix} X_k^T & X_{l}^T  \end{bmatrix} \begin{bmatrix} y_{kj}^{'} \\ y_{lj}^{'} \end{bmatrix}
$$

所有神经元在单次(第k次)输入下的参数更新量为：

$$
X_k^T \begin{bmatrix} y_{k1}^{'} & y_{k2}^{'}  & \dots &  y_{kC}^{'} \end{bmatrix} = X_k^T \cdot y_k^{'} 
$$

所有神经元在多次(N次)输入时的参数更新量为：

$$
\begin{bmatrix} X_1^T & X_2^T & \cdots & X_N^T \end{bmatrix} \begin{bmatrix} y_{11}^{'} & y_{12}^{'} & \cdots & y_{1C}^{'} \\ y_{21}^{'} & y_{22}^{'} & \cdots & y_{2C}^{'} \\ \vdots & \vdots & \ddots& \vdots \\ y_{N1}^{'} & y_{N2}^{'} & \cdots & y_{NC}^{'} \end{bmatrix} =    \begin{bmatrix} X_1^T & X_2^T & \cdots & X_N^T \end{bmatrix}  \begin{bmatrix} y_{1}^{'}  \\ y_{2}^{'}  \\ \vdots\\ y_{N}^{'} \end{bmatrix}
$$

其中

$$
X_k = \begin{bmatrix} X_{k1} & X_{k2} & \dots & X_{kD} \end{bmatrix} \text{表示该layer第k次输入}
$$ 

$$
y_k^{'} = \begin{bmatrix} y_{k1}^{'} & y_{k2}^{'} & \dots & y_{kC} \end{bmatrix}  \text{表示该layer第k次输出}
$$



假设我们的输入为$$ X \in R^{N \times D}$$, 参数$$W \in R^{D \times C}$$， 此时的输出为$$y \in R^{N \times C}$$

$$
dW = X^T \cdot dOut
$$

此处的dOut表示反向传播时后一层对本层输出的偏导，其大小与该层输出大小相同，为 $$N \times C$$, 而$$X^T$$的大小为  $$D \times N$$, 所以dW的大小为 $$D \times C$$, 可以发现 dW的大小与W的大小相同，我们这种矩阵的表示方式仅仅是为了计算方便，其本质还是对单个神经元参数的求导。

# 三、反向传播的总结

1. 反向传播实际上只是求单个神经元的偏导，矩阵的表示方式是为了方便我们一次更新多个神经元的参数，且同时可以进行多次输入时的更新
2. 单个神经元的多次输入下的更新可以线性叠加，每次更新只与输入有关， 系数只与后一层对该神经元的偏导有关
3. 所以神经元在单次输入下的更新全部只与该次 输入 成正比，系数只与后一层对神经元的偏导有关
4. 对于任意反向传播的推导， 只需要了解该层上某一个神经元即可。重点掌握两部分，该神经元的输出对输入的偏导，以及后一层对该神经元输出的偏导。之后用矩阵的形式将其表示成多次输入下的所有神经元更新即可。

# 四、例子

我们以神经网络中softmax的推导为例，来说明这篇文章中的主要内容：

<img src = "/images/PRML/backprop_essential/softmax.png">

假设该softmax层(包括一个全连接层)的输入有D个神经元，全连接层有C个神经元，最后经过一个softmax函数输出C个实数。

我们假设有N次输入，输入表示成 $$ X \in R^{N \times D} $$, 全连接层的参数表示为 $$W \in R^{D \times C}$$, 全连接层的输出表示成$$y \in R^{N \times C}$$, 而经过softmax的输出为 $$f \in R^{N \times C}$$. 

而第k次训练的sample的ground truth label表示为 $$t_k$$

根据前文，我们仅需要知道后一层对该层输出的偏导即可。


因为一般将softmax层作为最后一层，之后计算损失函数。那么我们可以看一下损失函数对于全连接层的第j个神经元的偏导即可。

因为

$$
L = -log(\frac{e^{f_{t_k}}}{\sum_{j = 1}^{N} e^{f_j}})
$$

那么我们有

$$
\frac{d L}{d y_{kj}} = y_{kj} - \delta(j = t_k)
$$
其中， 当$$j = t_k$$时， $$\delta(j = t_k)$$为1， 否则为0


那么我们根据前文所述，只用将dOut表示成 $$y- \delta{j =t},即在每次输入的时候，在第$$t_k$个神经元的输出上减去1即可，而全连接层的推到与前文完全相同。


所以softmax的反向传播可以如下所求：


```python
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorizted version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #scores -= np.max(scores, axis = 1)
  scores = np.exp(scores)
  total_scores_per_input = np.sum(scores,axis = 1).reshape(num_train,-1)

  scores /=total_scores_per_input

  correct_class_score = scores[range(num_train),y]

  loss_cost = -np.sum(np.log(correct_class_score))/num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost+loss_reg


  target = np.zeros_like(scores)
  target[range(num_train),y] = 1
  #or
  #dscores = p
  #sscores[range(num_train),y]-=1

  dW = X.T.dot(scores- target)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
```


```python

```
