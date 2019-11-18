---
layout: post
title: Go through Attention mechanism
author: Gin 
excerpt_separator: <!--more-->
tag: Deep Learning, Attention
toc: true
categories: [Deep Learning,Tech]
---


最近的工作内容跟Attention机制紧密结合，感觉有必要仔细梳理一下这块内容


## Origin Bahdanau attention

最先使用attention的一篇文章是[Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)

其基本结构是在原encoder-decoder-model的基础上, 增加了动态数量的encoder-embedding，让模型的表达能力更强了

<!--more-->

> In the Encoder–Decoder framework, an encoder reads the input sentence, a sequence of vectors x=(x1,···,xTx),in to avectorc.2 The most commonapproach is to use an RNN such that 
> <br>$$h_{t} = f(x_{t}, h_{t-1})$$
> <br>$$ c = q({h_{1}, ..., h_{T_{x}}}) $$
> <br>$$ s_{i} = r(s_{i-1}, y_{i-1}, c_{i}) $$
> <br>$$ y_{i} = g(y_{i-1}, s_{i}, c_{i}) $$ 

这里的f就是encoder，原文中使用的是bi-RNN结构, q是Attention，C是encoder-embedding经过Attention之后当前timestep的Context Vector, g是Decoder，一般而言也是一个RNN结构后面再接一些postNet，$$s_{i}$$ 是第i个timestep，Decoder中RNN的hidden, $$y_{i}$$ 是第i个timestep的输出，这里的$s_{i}$也是当前timestep得到的, RNN输入的是$s_{i-1}, y_{i-1}, c_{i}$, 输出是$s_{i}$, Decoder的输出是$y_{i}$

$$ C_{i} = \sum_{j=1}^{T_{x}}a_{ij}h_{j}$$ 

计算第i个timestep的context vector

$$a_{ij}$$ 表示第i个timestep，context vector与第j个encoder embedding的相关程度

$$a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_{x}}exp(e_{ik})} $$

其中$$e_{ij}$$ 表示初步计算的相关程度, 在这之后过一层softmax得到$$a_{ij}$$

$$e_{ij} = a(s_{i-1}, h_{j})$$

这里的$$a$$是alignment model，论文中使用的是feedforward network，在后来的应用中，甚至可以直接简化为$$s_{i-1}$$与$$h_{j}$$的点积


## Content-based attention

Tacotron的原始论文使用了这种Attention机制，首次提出这种机制的是这篇文章[Grammar as a foreign Language](https://arxiv.org/abs/1412.7449)

他在Bahdanua Attention的基础上做了点“细微的工作” 个人觉得改动非常小，但是不知道为什么Tacotron会采用这种形式的Attention

> An important extension of the sequence-to-sequence model is by adding an attention mechanism. We adapted the attention model from [2] which, to produce each output symbol Bt, uses an attention mechanism over the encoder LSTM states. Similar to our sequence-to-sequence model described in the previous section, we use two separate LSTMs (one to encode the sequence of input words Ai, and another one to produce or decode the output symbols Bi). Recall that the encoder hidden states are denoted (h1, . . . , hTA ) and we denote the hidden states of the decoder by (d1, . . . , dTB ) := (hTA+1, . . . , hTA+TB ).

废话不多说，直接看公式

> 
  $$ 
  \begin{align*}
  & u_{i}^{t} = v^{T}tanh(W_{1}^{'}h_{i} + W_{2}^{'}d_{t}) \\
  & a_{i}^{t} = softmax(u_{i}^{t}) \\
  & d_{t}^{'} = \sum_{i=1}^{T_{A}}a_{i}^{t}h_{i} \\ 
  \end{align*}
  $$

上面是原论文的公式，看的我没头没脑，改写了一下保持和原Bahdanau Attention的标记一致

$$
\begin{align*}
& e_{ij} = V^{T}tanh(W_{1}^{'}h_{j}+W_{2}^{'}s_{i-1}) \\
& a_{ij} = \frac{e_{ij}}{\sum_{k=1}^{T_{A}}e_{ik}} \\
& c_{i} = \sum_{j=1}^{T_{A}}a_{ij}h_{j}
\end{align*}
$$

这样看上去顺眼多了

> The vector v and matrices W1′ , W2′ are learnable parameters of the model. The vector ut has length TA and its i-th item contains a score of how much attention should be put on the i-th hidden encoder state hi. These scores are normalized by softmax to create the attention mask at over encoder hidden states. In all our experiments, we use the same hidden dimensionality (256) at the encoder and the decoder, so v is a vector and W1′ and W2′ are square matrices. Lastly, we concatenate d′t with dt, which becomes the new hidden state from which we make predictions, and which is fed to the next time step in our recurrent model.

接着文章中说，会把得到的context vecotr和上一个timestep的输出concatenate起来，然后作为decoder的输入得到当前timestep的输出, 也就是两个[1, 256]的的vector被concate成一个[1, 512]的vector, 作为Decoder LSTM当前timestep的输入，然后得到Decoder LSTM的当前timestep输出的hidden也就是$s_{i}$

整个结构跟原Bahdanua Attention没啥大的区别，唯一值得注意的地方就是计算energy的那个公式

$$
\begin{align*}
& e_{ij} = V^{T}tanh(W_{1}^{'}h_{j}+W_{2}^{'}s_{i-1}) \\
\end{align*}
$$

原来的alignment model是一个DNN，这里引入了一个新的可学习参数$V$，虽然不太清楚为啥，但是看上起并没有什么大不了的

## location-sensitive Attention

Tacotron2 中使用了这种attention机制, 提出这种Attention机制的原始论文是[Attention-based models for speech recognition](https://arxiv.org/abs/1506.07503), 又是出自Bahdanua之手，不过这次，他把Attention应用到了Speech recongnition上

> We introduce extensions to attention-based recurrent networks that make them applicable to speech recognition. Learning to recognize speech can be viewed as learning to generate a sequence (tran- scription) given another sequence (speech). From this perspective it is similar to machine translation and handwriting synthesis tasks, for which attention-based methods have been found suitable [2, 1]. However, compared to machine translation, speech recognition principally differs by requesting much longer input sequences (thousands of frames instead of dozens of words), which introduces a challenge of distinguishing similar speech fragments2 in a single utterance

首先他做了一个实验，用原始的Bahdanua Attention建立语音识别模型，发现在短序列的数据集上效果还行，但是长数据集就错的非常惨，接着他说，证据表明模型之所以会这样表现，是因为其试图通过绝对位置去识别特定的内容，这种策略在短序列上还不会相差太多，但是到了长数据集上就行不通了

> We provide evidence that this model adapted to track the absolute location in the input sequence of the content it is recognizing, a strategy feasible for short utterances from the original test set but inherently unscalable.

然后他就突发奇想，如果让Attention在学习alignment的时候能够参考前面alignment的结果，会不会就能解决这个问题呢？

> This is achieved by adding as inputs to the attention mechanism auxiliary convolutional features which are extracted by convolving the attention weights from the previous step with trainable filters.

话休烦絮，上公式

encoder 的output embedding依然表示为

$$h = (h_{1}, ...., h_{L})$$

当然，encoder也还是熟悉的配方，以RNN为主的结构, 这里用的是Bi-RNN

接着到了Attention部分

$$
\begin{align*}
& a_{i} = Attend(s_{i-1}, a_{i-1}, h) \\
& g_{i} = \sum_{j=1}^{L}a_{ij}h_{j} \\
& y_{i} = Generate(s_{i-1}, g_{i})
\end{align*}
$$

有一个地方我怀疑是他写错了

$$
s_{i} = Recurrency(s_{i-1}, g_{i}, y_{i})
$$

这里我觉得应该是$y_{i-1}$, 因为$y_{i}$是第i个timestep的输出，怎么会在计算第i个timestep的$s_{i}$的时候用到呢

$$
s_{i} = Recurrency(s_{i-1}, g_{i}, y_{i-1})
$$

这样我觉得是合理的

Badhanua在这里建设性的提出了一个定义, 只要是

$$
a_{i} = Attend(s_{i-1}, h)
$$

的Attention结构，叫做content-based Attention
然后，

$$
a_{i} = Attend(a_{i-1}, h)
$$

这样的结构，叫做location-based Attention

文中使用的结构，

$$
a_{i} = Attend(s_{i-1}, a_{i-1}, h)
$$

被称之为，hybrid-based Attention

所以按这个定义来的话，上面提到的两种Attention都属于content-based attention

接着他解释了，为什么content-based attention有局限性

$$
\begin{align*}
& e_{ij} = a(s_{i-1}, h_{j}) \\
& a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_{x}}exp(e_{ik})}
\end{align*}
$$

如果两个encoder embedding很相似，那么计算出来的energy就会很像素，那么模型就会同等对待这两个embedding, 但是，这两个embedding出现在不同的位置，所代表的含义是不一样的

> The main limitation of such scheme is that identical or very similar elements of h are scored equally regardless of their position in the sequence. This is the issue of “similar speech fragments” raised above. Often this issue is partially alleviated by an encoder such as e.g. a BiRNN [2] or a deep convolutional network [3] that encode contextual information into every element of h . However, capacity of h elements is always limited, and thus disambiguation by context is only possible to a limited extent.


如果是content-based attention，计算energy的方法是

$$
e_{ij} = w^{T}tanh(Ws_{i-1} + Vh{j} + b)
$$

文中使用的计算方法，引入了上一个timestep alignment的信息，具体方法是用
k个卷积核在上一个timestep的alignment上卷过去，每个position会产生k个vector

$$
f_{i} = F*a_{i-1}
$$

然后加入这个信息去计算energy

$$
e_{ij} = w^{T}tanh(Ws_{i-1} + Vh_{j} + Uf_{ij} + b)
$$


除此之外，文章还讨论了怎么去调整，计算$a_{ij}$时候的scale方法

如果希望模型的attention集中于少数h，那么可以做sharpening

1)  $$
a_{ij} = \frac{exp(\beta e_{ij})}{\sum_{j=1}^{L}(exp(\beta e_{ij}))}
$$

这里的$\beta > 1$这样调整temperature可以增大区分度

2) 只选择k个值最高的energy做re-normalize

如果希望模型平均attention则采取另一种策略，smoothing
文中采取的策略是，将softmax中的exp函数替换为sigmoid函数, 这样使norm过的值大小相差不会太多

$$
a_{ij} = \frac{sigmoid(e_{ij})}{\sum_{j=1}^{L}(sigmoid(e_{ij}))}
$$

## GMM-attention

首次提出GMM-attention机制的是，[Generating sequences with recurrent neural networks](https://arxiv.org/abs/1308.0850), 这篇文章发表于2013年，甚至早于Bahdanau等人提出Attention的概念，并且在这篇文章中，作者不自知的使用了后来Kaiming He在2015年提出的Residual connection，在这篇文章中，作者称之为“Skip connection”，他偶然发现添加这样的连接有助于深层神经网络的收敛.

> Note the ‘skip connections’ from the inputs to all hidden layers, and from all hidden layers to the outputs. These make it easier to train deep networks, by reducing the number of processing steps between the bottom of the network and the top, and thereby mitigating the ‘vanishing gradient’ problem.

可惜的是他并没有就此深入研究这个看似简单的技巧。

提出GMM-Attention的场景是，用给的的text，生成对应的手写字母，这个任务的主要难点在于一个字母对应的手写字母是不定长的，并且只有当生成完毕之后，我们才能找到他们之间的对应关系，作者在这里创造性的使用了动态网络来决定当前需要生成哪一个字母的手写体，这在当时尚没有Attention的概念时是非常难能可贵的。

论文的网络结构如下

![image.png](https://i.loli.net/2019/08/10/vCKTmgVhf4QoYD3.png){:class="center" width="50%"}


给定的训练对是，长度为$U$串c，以及长度为$T$的手写字母sequence，x，GMM-Attention的计算公式如下

$$
\begin{align*}
& \theta(t,u) = \sum_{k=1}^{K}\alpha_t^k exp(-\beta_t^k(\kappa_t^k - u)^2) \\
& w_t = \sum_{u=1}^{U}\theta(t, u)c_u
\end{align*}
$$

这里的假设是，每个timestep的要写的character由一个window决定，而window则由$K$个Gaussian分布决定,其中$\theta(t,u)$是$c_u$在第t个timestep的权重，也就相当于Attention机制中的align，此外跟window相关的三个variable，$\kappa_t^k$是分布的mean，相当于window的中心，$\kappa_t^k$跟某个的字母位置越接近，说明当前timestep输出这个字母的可能性越大, 反映在公式中就是，当$\kappa_t^k$与$u$相等时，指数部分的结果为0，此时$\theta(t, u)$取值最大。$\beta_t^k$表示window的大小，其实这里有点像前面提到的sharpen技巧，$\beta_t^k$的值越小，window的size越大，那么分散到window内每一个元素的注意力就越平均，如下图所示

![image.png](https://i.loli.net/2019/08/10/bH9wGlCvVpmuNic.png){:class="center"}

$\alpha_t^k$表示当前Gaussian分布的权重, 计算出$\theta(t, u)$之后接着就是算context vecotr了，$w_t$就表示当前timestep，Attention的结果

回过头来，每个分布的$\alpha \beta \kappa$是怎么来的呢，这里是通过第一层LSMT的hidden过一层dense得到的

$$
\begin{align*}
& (\hat{\alpha_t}, \hat{\beta_t}, \hat{\kappa_t}) = W_{h^1p}h_t^1 + b_p \\
& \alpha_t = exp(\hat{\alpha_t}) \\
& \beta_t = exp(\hat{\beta_t}) \\
& \kappa_t = \kappa_{t-1} + exp(\hat{\kappa_t})
\end{align*}
$$

$\alpha_t \beta_t \kappa_t$都是长度为K的vecotr，$W_{h^1p}$是shape为$[3K, h]$, $h$是第一层LSTM输出hidden的大小，这里是一个典型的location-based attention机制, 因为在计算$\theta(t, u)$时仅仅使用到来上一个timestep的$\kappa_{t-1}$, 当然也可以在$(\hat{\alpha_t}, \hat{\beta_t}, \hat{\kappa_t})$的时候，引入额外的信息比如含有key信息的$C$, 这样就成了Hybrid-attention, 当然这是题外话，暂且不表。

下面来谈一谈GMM-attention的特点，以及为什么很多tacotron的实现使用了这种Attention。

由于多个Gaussian distribution的混合使得这样的attention机制学习alignment的能力非常强，能够习得非常复杂的alignment方式，而且因为有$\kappa_t = \kappa_{t-1} + exp(\hat{\kappa_t})$这样的操作，使得每一个timestep当前关注的window位置都只会往前，保证了monotonicity, 在tacotron的场景下，每个frame关注的phone只可能是前一个frame关注的phone, 或者是前一个frame关注phone的后一个phone, 这是由声学模型的特征决定的，因为frame划分的粒度非常细，一般为12.5ms或者5ms，当前frame发音受其他phone的影响很小, 这是tacotron应用场景和生成手写体字母这个任务相似的地方，可以说GMM-attention最大的助益就是提供了单调性。

但是光是保证单调性我觉得是不够的，因为当$exp(\hat{\kappa_t})$很大的时候，会一下子跳过多个character，导致丢失了许多应该捕捉的信息，我个人觉得这里的exp改成sigmoid会更好一点，因为需要控制$\kappa_t$的增长, 这样使之小于等于1就不会发生一下子跳过很多character的问题。

## Forward attention

读mozila的tacotron代码时发现的这个技巧，原始paper是中科大的一帮人写的,[Forward attention in Sequence-to-sequence Acoustic Modeling](https://arxiv.org/abs/1807.06736). 与其说这是attention机制不如说这是attention机制的一种后处理方式。

先看看content-based attention是怎么样的

$$
\begin{align*}
& e_{t,n} = w*tanh(W_1*h_n + W_2*q_t + b) \\
& y_{t, n} = \frac{exp(e_{t, n})}{\sum_{\hat{n}=1}^{N}e_{t, n}} \\
& c_t = \sum_{n=1}^{N}y_{t, n}h_n

\end{align*}
$$

其思想引入了一个假设，假设${\pi_1, \pi_2, .., \pi_T}$是T个timestep所选取的phone idx的集合(这里我们称之为alignment path)

在Acoustic model这样的场景下，比较合理的路径, 需要满足这样的条件$\pi_i = \pi_{i-1}$或者$\pi_i = \pi_{i-1} + 1$

![image.png](https://i.loli.net/2019/08/10/rFs8iGauvMkIg5E.png)

我们认为每个timestep选取的$\pi$都是独立同分布的，那么形成路径$\pi_{1:t} = {\pi_1,...,\pi_t}$的概率是

$$
p(\pi_{1:t}|x, q_{1:t}) = \prod_{\hat{t} = 1}^{t}p(\pi_{\hat{t}} | x, q_{\hat{t}}) = \prod_{\hat{t} = 1}^{t}y_{\hat{t}}(\pi_{\hat{t}})
$$

这里再引入一个定义，如果$\mathcal{p}$代表所有${\pi_0, \pi_1,...,\pi_t}$, 的合理路径, 其中$\pi_0 = 1, \pi_t = n$, 其他时刻的$\pi$可取满足上述合理假设的任意值, 在这样的条件下，

$$
\alpha_t(n) = \sum_{\pi_{1:t}\in\mathcal{p}}\prod_{\hat{t} = 1}^{t}y_{\hat{t}}(\pi_{\hat{t}})
$$

根据$\alpha_t{n}$的性质有

$$
\alpha_t(n) = (\alpha_{t-1}(n) + \alpha_{t-1}(n-1))*y_t(n)
$$

这里的$\alpha_t(n)$表示的是形成合理的${\pi_1,...,\pi_t}, \pi_t = n$路径的概率, $y_t{n}$表示的是第t个timestep选择第n个phone的概率, 这个等式看上去没毛病, 通过初始化$\alpha_0(0) = 1, \alpha_0(1) = 0, ..., \alpha_0(N) = 0$, 我们可以在每个timestep t都得到$\alpha_t{1}, ..., \alpha_t{n}$

然后把上面一系列数scale一下, 使之和为1

$$
\begin{align*}
& \hat{\alpha_{t}(n)} = \frac{\alpha_{t}(n)}{\sum_{i=1}^{N}\alpha_{t}(i)} \\
&  c_t = \sum_{n=1}^{N}\alpha_{t}(n)x_n
\end{align*}
$$


$\hat{\alpha_{t}}(1:n)$ 就是新形成的alignment，重新分布的alignment比原始alignment更能够满足应用场景的要求, 在合理路径的假设下单调和逐步两个条件都满足

在此基础上，作者又提出了一种Transition Agent的机制，就是让网络自己决定当前timestep应该前进还是停留

$$
\begin{align*}
& \alpha_t(n) = (u_{t-1}\alpha_{t-1}(n) + (1 - u_{t-1})\alpha_{t-1}(n-1))*y_t(n) \\
& u_t = DNN(c_t, q_t, o_{t-1})
\end{align*}
$$

根据论文的实验结果，forward attention不仅能显著降低，丢字、叠字、发音不清的情况的出现概率，而且能加速alignment的形成

##  Step-wise monotonic Attention

也许是受Forward attention的启发，北航和微软合著了一篇[robust sequence-to-sequence acoustic modeling with stepwise monotonic attention for neural tts](https://arxiv.org/abs/1906.00672), 但是他们不承认是脱胎于Forward attention，觉得应该是Monotonic attention的进化版

其思想非常简单, 每一个timestep要么前进一个phone，要么停留在当前phone, 每一个timestep取第j个phone的概率等于，上一个timestep取第j个phone的概率乘上当前timestep 第j个phone被选择的概率, 加上上一个timestep取第j-1个phone乘上当前timestep不选择第j-1个phone的概率，不选择第j-1和phone那么只能前进一个phone，也就是选择第j个phone, 看公式更加清晰

$$
\begin{align*}
& \alpha_{ij} = \alpha_{i-1, j-1}*(1 - p_{i, j-1}) + \alpha_{i-1, j} * p_{i, j} \\
& p_{i, j} = sigmoid(e_{i, j})
\end{align*}
$$
