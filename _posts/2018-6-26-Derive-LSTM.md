---
layout: post
title: Derive LSTM
author: Gin 
excerpt_separator: <!--more-->
tag: Deep Learning, LSTM
categories: [Deep Learning,Tech]
---

[参考文献](https://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)


>We could also do something more interesting: we can let the RNN decide when its ready to move on to the next input, and even what that input should be. This is similar to how a human might focus on certain words or phrases for an extended period of time to translate them or might double back through the source. To do this, we use the RNN’s output (an external action) to determine its next input dynamically. For example, we might have the RNN output actions like “read the last input again”, “backtrack 5 timesteps of input”, etc. Successful attention-based translation models are a play on this: they accept the entire English sequence at each time step and their RNN cell decides which parts are most relevant to the current French word they are producing.

![RNN overview](https://r2rt.com/static/images/NH_VanillaRNNcell.png)

***这里解释为什么RNN需要上一个timestep的output， 上一个output既是一种结果，也是对网络本身的反馈，或者说根据上一个output哪里做的不好，那么在当前timestep里就需要对这里缺失的信息进行加强。这种解释我觉得是很合理的。***

<!--more-->
>Note how the prior state vector is the same size as the current state vector. As discussed above, this is critical for composition of RNN cells. Here is the algebraic description of the vanilla RNN cell:
st=ϕ(Wst−1+Uxt+b)st=ϕ(Wst−1+Uxt+b)
where:
* ϕϕ is the activation function (e.g., sigmoid, tanh, ReLU),
* st∈Rnst∈Rn is the current state (and current output),
* st−1∈Rnst−1∈Rn is the prior state,
* xt∈Rmxt∈Rm is the current input,
* W∈Rn×nW∈Rn×n, U∈Rm×nU∈Rm×n, and b∈Rnb∈Rn are the weights and biases, and
* nn and mm are the state and input sizes.
 We’ll see below that although information morphing was not among the original motivations for introducing LSTMs, the principle behind LSTMs happens to solve the problem effectively.


***所以LSTM设计的初衷并没有考虑到解决information morphing问题，那么这个设计的motivation是什么呢？***

>In fact, the effectiveness of the residual networks used by He at al. (2015) is a result of the fundamental principle of LSTMs.

***这一点我也颇为认同，LSTM的设计存在一条identity connect 但是没有数学推导，ResNet的shortcut连接比较直观，也能给出明确的数学推导。***

>If they vanish, it’s difficult for us to learn long-term dependencies, since backpropagation will be too sensitive to recent distractions. This makes training difficult.

***long-term dependency在存在gradient vanishing的情况下，变得影响很小，所以对当前time Step output来说如何获取long-term information 是关键，因为在序列模型中，远距离信息与近距离信息都很重要，LSTM的设计应该就是从这一点出发的。
   sup指的是最小下界***

>维基百科：
依照范数的定义，一个从$${\displaystyle {\mathcal {M}}_{m,n}(\mathbb {K} )}$$映射到非负实数的函数$${\displaystyle \|\cdot \|}$$满足以下的条件：
* 严格正定性：对任意矩阵$${\displaystyle A\in {\mathcal {M}}_{m,n}(\mathbb {K} )}$$，都有$${\displaystyle \|A\|\geq 0}$$，且等号成立当且仅当$${\displaystyle A=0}$$；
* 线性性：对任意系数$${\displaystyle \alpha \in \mathbb {K} }$$、任意矩阵$${\displaystyle A\in {\mathcal {M}}_{m,n}(\mathbb {K} )}$$，都有$${\displaystyle \|\alpha A\|=|\alpha |\|A\|}$$；
* 三角不等式：任意矩阵$${\displaystyle A,B\in {\mathcal {M}}_{m,n}(\mathbb {K} )}$$，都有$${\displaystyle \|A+B\|\leq \|A\|+\|B\|}$$。则称之为$${\displaystyle {\mathcal {M}}_{m,n}(\mathbb {K} )}$$上的一个矩阵范数。
此外，某些定义在方块矩阵组成空间$${\displaystyle {\mathcal {M}}_{n}(\mathbb {K} )}$$上的矩阵范数满足一个或多个以下与的条件：
* 一致性：$${\displaystyle \|AB\|\leq \|A\|\|B\|}$$；
* 共轭转置相等条件：$${\displaystyle \|A\|=\|A^{*}\|}$$。其中$${\displaystyle A^{*}}$$表示矩阵$${\displaystyle A}$$的共轭转置（在实矩阵中就是普通转置）。
一致性特性（consistency property）也称为次可乘性（sub-multiplicative property）。

|Δst+1|=|[ϕ′(c)]WΔst|
            ≤‖[ϕ′(c)]‖‖W‖|Δst|
            ≤γ‖W‖|Δst|
            =‖γW‖|Δst|.
范数具有这样的性质

>By expanding this formula over kk time steps we get |Δst+k|≤‖γW‖k|Δst||Δst+k|≤‖γW‖k|Δst| so that:
|Δst+k||Δst|≤‖γW‖k.|Δst+k||Δst|≤‖γW‖k.
Therefore, if ‖γW‖<1‖γW‖<1, we have that |Δst+k||Δst||Δst+k||Δst| decreases exponentially in time, and have proven a sufficient condition for:
limk→∞Δst+kΔst=0.limk→∞Δst+kΔst=0.
When will ‖γW‖<1‖γW‖<1? γγ is bounded to 1414 for the logistic sigmoid and to 1 for tanh, which tells us that the sufficient condition for vanishing gradients is for ‖W‖‖W‖ to be less than 4 or 1, respectively.


>An immediate lesson from this is that if our weight initializations for WW are too small, our RNN may be unable to learn anything right off the bat, due to vanishing gradients. Let’s now extend this analysis to determine a desirable weight initialization.

***这就是为什么权重初始化会对网络很重要，因为权重太小的话bottom层的weights根本不会得到更新，或者说较top层的weights可以更新但是会很慢，因此反映的loss就会stuck***

>RNNs have it worse, because unlike for feedforward nets, the weights in early layers and later layers are shared. This means that instead of simply miscommunicating, they can directly conflict: the gradient to a particular weight might be positive in the early layers but negative in the later layers, resulting in a negative overall gradient, so that the early layers are unlearning faster than they can learn. In the words of Hochreiter and Schmidhuber (1997): “Backpropagation through time is too sensitive to recent distractions.”

***由于共享参数，但是前后timestep学习到的特征又可能是迥异的，因此很容易发生冲突，这也是RNN比较难训练的地方，比如，我是XXX， 前面学习到的可能是名词，“我是李雷”，“我是小明”，但是后面可能就是“我是相信他的”，变成了动词，很明显，不同性质的词在空间中是差别很大的，这是相关的weights就会conflict，前面学习到的特征会被后面学习的覆盖掉***


>How can we protect the integrity of messages? This is the fundamental principle of LSTMs: to ensure the integrity of our messages in the real world, we write them down. 

***LSTM的motivation是保存输入信息的完整性，保存的媒介就是hidden state***


>The answer, quite simply, is to avoid information morphing: changes to the state of an LSTM are explicitly written in, by an explicit addition or subtraction, so that each element of the state stays constant without outside interference: “the unit’s activation has to remain constant … this will be ensured by using the identity function”.

***但是LSTM里面没有用到Identity function，它是怎么保证梯度传播的完整性的？
This is the fundamental challenge of LSTMs: Uncontrolled and uncoordinated writing causes chaos and overflow from which it can be very hard to recover.
因此需要“遗忘”一些东西，这个怎么想出来的？***

>According to the early literature on LSTMs, the key to overcoming the fundamental challenge of LSTMs and keeping our state under control is to be selective in three things: what we write, what we read (because we need to read something to know what to write), and what we forget (because obselete information is a distraction and should be forgotten).

***这里有点意思，需要遗忘的是什么样的信息？首先，需要记住的是能够描述最终分布的参数，那么反过来讲需要遗忘的就是不能描述最终分布的参数，怎么判断其是否能描述，就是根据训练的样本，能够很好的和样本相匹配的参数应该被记录下来，反之，与训练样本发生冲突的则需要被遗忘，因为这说明之前的描述方式不对，需要再考虑多一点的样本才能更好的描述。***

>First form of selectivity: Write selectively.
To get the most out of our writings in the real world, we need to be selective about what we write; when taking class notes, we only record the most important points and we certainly don’t write our new notes on top of our old notes. In order for our RNN cells to do this, they need a mechanism for selective writing.

***输入门的设计初衷，但是觉得有点“从结果解释原因”的意思***

>Hochreiter and Schmidhuber describe this as “output weight conflict”: if irrelevant units are read by all other units at each time step, they produce a potentially huge influx of irrelevant information. Thus, the RNN must learn how to use some of its units to cancel out the irrelevant information, which results in difficult learning.

***输出门的由来，在记录的信息中，并不是所有的东西都对当前timestep有用，所以需要控制输出的东西***


>Third form of selectivity: Forget selectively.
In the real-world, we can only keep so many things in mind at once; in order to make room for new information, we need to selectively forget the least relevant old information. In order for our RNN cells to do this, they need a mechanism for selective forgetting.

***遗忘门，随着训练的进行只保留有效的信息。***


>With that, there are just two more steps to deriving the LSTM:
1. we need to determine a mechanism for selectivity, and
2. we need to glue the pieces together.

***上面讲述了3个门背后的思想，下面就是讲如何实现这些机制，***


>The fundamental principle of LSTMs says that our write will be incremental to the prior state; therefore, we are calculating ΔstΔst, not stst. Let’s call this would-be ΔstΔst our candidate write, and denote it s̃ ts~t.


***LSTM运行的机制***

***1.*** 

$$\tilde{s_t} = \phi(W(o_t \odot s_{t-1}) + Ux_t + b$$

***首先，需要先从上一个hidden state中“读”出一些信息，综合当前timestep的输入构成一个待修改的hidden state，这些相当于是当前timestep的全部输入信息。***

***2.***
 
$$s_t = f_t \odot s_{t-1} + i_t \odot \tilde{s}_t$$

***然后，我们需要更新这个共享hidden state，如何更新？上一个hidden state需要遗忘一些东西，然后加上当前shallow hidden state需要被记住的一部分，这里有一点值得注意的是，hidden state是全程被共享的，所谓的长短时记忆，长期记忆的就是hidden state ，遗忘门也是作用在hidden state上，而输入门则是作用在shallow hidden state上，随后两者通过加法门组成新的hidden state；这里有一点像ResNet，bottom layer学习训练数据分布的主体部分，后续的layer通过不断学习残差逐步逼近模型；LSTM则使用shallow hidden state学习残差，并且是有选择的学习，因为它还使用了input gate控制残差的输入。***

***3. 说一说遗忘门***

***序列模型的一个特点就是，一个sequence的前一部分不能反映数据的整体分布，甚至与整体分布是完全相反的，因此在用sequence feed模型的时候，模型会先学习前一部分的特征，但是这时候学习到的东西会和后面的数据conflict，例如情感判断“这部电影好看”，“这部电影好看的只剩下演员了”，前者表达正面情绪，后者表达负面情绪，前一部分的表达是相同的，当模型学习到“这部电影好看”的时候，会输出较大的正面情绪数值，这是因为前面的一句话对weights产生了影响，但实际上当前训练的这段话是negative的，loss发生了严重不匹配，所以weights更新的时候会与前一个case冲突，极端情况下会导致loss一直不收敛；这个时候遗忘门的重要性就体现出来了，它会让模型忘记前面学习到的conflict weights，重新使用shallow hidden state学习到的新weights以适应当前的sequence。***

***4. GRU的改进***

***GRU对vanilla LSTM的一大改动就是“丢弃”了遗忘门，他的hidden state更新是这样的：***

$$s_t = (1-i_t) \odot s_{t-1} + i_t \odot \tilde{s}_t$$

***根据我上面的理解这个公式直观的表达就是，更新的shallow hidden state被重视的部分在原hidden state里面就要被轻视，在shallow hidden state里面被轻视的，在原hidden state里面就要被重视，但是这样也有一点不对，你遗忘的东西并不一定是你当前需要记住的，你当前需要记住的也不一定就要丢弃以前记住的。
虽然说GRU的改进使得模型的参数量减少了1/3，训练和推理更快了，但是我不觉得这样的模型具有很好的表达能力。***


>This works because it turns stst into an element-wise weighted average of st−1st−1 and s̃ ts~t, which is bounded if both st−1st−1 and s̃ ts~t are bounded. This is the case if we use ϕ=tanhϕ=tanh (whose output is bound to (-1, 1)).

***终于看到了一点为什么需要使用tanh的解释，因为输出的hidden state需要被bounded。***

***GRU***

$$
\begin{equation}
\begin{split}
r_t &= \sigma(W_rs_{t-1} + U_rx_t + b_r) \\
z_t &= \sigma(W_zs_{t-1} + U_zx_t + b_z) \\
\\
\tilde{s_t}& = \phi(W(r_t \odot s_{t-1}) + Ux_t + b)\\
s_t &= z_t \odot s_{t-1} + (1 - z_t) \odot \tilde{s}_t
\end{split}
\end{equation}
$$

***大体思路与LSTM相似，先读取上一个timestep的hidden state，和当前input构建read gate，使用read gate更新shallow hidden state，然后使用一个reset gate去更新当前的hidden state，这里的reset gate既充当遗忘门，又充当输入门，对比一下原始LSTM，read gate两者都存在，reset gate 的作用相当于input gate和forget gate，只是GRU里面用了一种加权平均的方法***

$$
\begin{equation}
\begin{split}
i_t &= \sigma(W_is_{t-1} + U_ix_t + b_i) \\
o_t &= \sigma(W_os_{t-1} + U_ox_t + b_o) \\
f_t &= \sigma(W_fs_{t-1} + U_fx_t + b_f) \\
\\
\tilde{s_t}& = \phi(W(o_t \odot s_{t-1}) + Ux_t + b)\\
s_t &= f_t \odot s_{t-1} + i_t \odot \tilde{s}_t
\end{split}
\end{equation}
$$


>Pseudo LSTM
 Instead, we pass the state through the squashing function every time we need to use it for anything except making incremental writes to it. By doing this, our gates and candidate write don’t become saturated and we maintain good gradient flow.

$$
\begin{equation}
\begin{split}
i_t &= \sigma(W_i(\phi(s_{t-1})) + U_ix_t + b_i) \\
o_t &= \sigma(W_o(\phi(s_{t-1})) + U_ox_t + b_o) \\
f_t &= \sigma(W_f(\phi(s_{t-1})) + U_fx_t + b_f) \\
\\
\tilde{s_t}& = \phi(W(o_t \odot \phi(s_{t-1})) + Ux_t + b)\\
s_t &= f_t \odot s_{t-1} + i_t \odot \tilde{s}_t\\
\\
\text{rnn}_{out} & = \phi(s_t)
\end{split}
\end{equation}
$$

***除了Increment那一步，在前面涉及前一个timestep Hidden state的地方都加上了一个squashing function(一般来说是tanh)但是这样做的目的是什么？***

>By doing this, our gates and candidate write don’t become saturated and we maintain good gradient flow.

***这里的饱和是当hidden state中的一个unit变成1的时候，就会一直是1，也就不会得到更新了，添加一个squashing function就是为了让它活性更大，训练时一直会更新。
这两种变形，目的都是给hidden state 加上 bound
最基础的LSTM是：***

$$
\begin{equation}
\begin{split}
i_t &= \sigma(W_ih_{t-1} + U_ix_t + b_i) \\
o_t &= \sigma(W_oh_{t-1} + U_ox_t + b_o) \\
f_t &= \sigma(W_fh_{t-1} + U_fx_t + b_f) \\
\\
\tilde{c_t}& = \phi(Wh_{t-1} + Ux_t + b)\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\\
\\
h_t &= o_t \odot \phi(c_t)\\
\\
\text{rnn}_{out} &= h_t
\end{split}
\end{equation}
$$

跟前面讲的LSTM结构看上去并无区别，但是这里的c和h要区分开，c是cell memory，h则是当前timestep的输出，
当前的门计算，使用的是上一个timestep的输出，这个与前面的LSTM并无区别，
计算 shadow cell memory时使用的是上一个timestep的输出
而另一种LSTM使用的是经过当前output门的hidden state
计算新cell memory的方法也并无不同
最后的输出则需要使用当前output门与squashing 作用后的cell memory得到。
