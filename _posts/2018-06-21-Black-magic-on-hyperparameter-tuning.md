---
layout: post
title: 调参黑魔法
author: Gin 
excerpt_separator: <!--more-->
tag: Linux, Shell
categories: [Deep Learning,Tech]
---
### Learning Rate

美国海军实验室的一篇report提到了一种cyclical learning rate的调参方法。

文章里说，lr过小会导致overfitting，这个的确是有可能的，较小的lr使得整体的weights更新很小，容易困在一个凹型区域当中，导致不断逼近局部最小值，这是过拟合的一种情况。

lr过大会导致function loss发散，因为没办法很精确的调节，一下左一下右，最终无法收敛，loss爆炸。

cyclical lr 是一种初步确定lr参数的方法，首先根据经验选定lr最小值和最大值，然后选择stepsize，从最小值慢慢增长到最大值，每一个step可以用一个iteration或者一个epoch来训练，可以线性式增长也可以离散式增长。

在开始训练时，由于lr很小，loss曲线会缓慢降低并有收敛的趋势，但是随着lr不断增大，loss最终就会爆炸，根据loss的图像，选择loss极小值处的lr作为lr range的最大值。

lr range最小值的确定就比较随意，通常是选择小于最大值3，4个数量级的数值。


<!--more-->
### Batch Size

这篇report提倡使用大batch.

>The constant number of epochs is inappropriate as it doesn’t account for the significant computational efficiencies of larger batch sizes so it penalizes larger batch sizes. On the other hand, a constant number of iterations favors larger batch sizes too much.

这里不是很明白，跑相同的epoch怎么就不能利用large batch size的计算优势了。跑相同iteration，large batch size是占优势的，这个没有异议。

>It is clear in Figure 6a that the larger batch sizes ran in fewer iterations. Not shown is that larger batch sizes had more 

epochs (TBS/epochs = 128/48, 256/70, 512/95, 1024/116).
这里说明了问题，在达到同样test accuracy的情况下，large batch size需要更多的epochs，因此如果固定epoch，large batch size的模型取得的效果肯定会比small batch size的差。

这个是什么原因呢？ 

$$
epochs = \frac{iterations \ast TBS}{total\ train\ data}
$$

在训练时间相同的情况下，大TBS的iteration数量较少，小TBS的iteration数量较多，但是这篇report给出的数据显示，这样计算下来的epochs数量明显是大TBS更多的，也就是说：

**在相同训练时间下，大TBS遍历训练数据的次数要比小TBS多**

见过更多次样本会导致模型性能提升吗？恐怕不见得

还有一个奇怪的现象：

>The effects of total batch size (TBS) on validation loss for the Cifar-10 	with resnet-56 and a 1cycle learning rate schedule. For a fixed 	computational budget, larger TBS yields higher test accuracy but smaller TBS has lower test loss.

相同计算时间下，大batch size准确率更高，而小batch size loss更小

但是有一点文章没有解释的是，为什么更大的batch size可以使用更大的学习率，我的理解是大的batch size在每次iteration时由于考虑的数据分布更稳定，因此更新weights具有相对的可靠性，而小batch size每个iteration被feed的数据量小了，数据分布可能差异较大，因此weights更新更容易发生冲突，使用较小的学习率有助于提升训练的整体稳定性

### Momentum

momentum 跟learning rate一样对weights的更新具有同样的影响力，下面的公式可以证明

$$
\theta_{iter+1} = \theta_{iter} - \varepsilon\delta L(F(x, \theta), \theta)\\ \  \\v_{iter+1} = \alpha v_{iter} - \varepsilon \delta L(F(x, \theta), \theta)\\ \theta_{iter+1} = \theta_{iter} + v
$$


可以看到$\varepsilon$和$\alpha$在权重更新公式中地位相同，因此momentum和learning rate其实具有同样重要的作用

这里作者说实验表明，range momentum并像range learning rate一样具有较好的筛选作用。文章中这里说在固定learning rate的情况下使用大的momentum（0.9 0.99）是一种变相的加速训练， 同时他推荐在训练中增大learning rate的同时适当减少momentum

![test image size]({{site.url}}/assets/images/2018_06_21_1.png){:class="center"}

可以看到在图中增大learning rate的同时减少momentum在训练初始阶段获得了较好的收敛速度以及较低的test loss，那么这样做的动机是什么？learning rate增大，weights更新的幅度大了，适当减少momentum用以对冲这种更新幅度，看起来比较合理，直观的看，momentum减少意味着指数平均 考虑过去的v值权重变小了，当前更新的值可以更快地得以表现，因此收敛速度变快了，但是为什么test loss的最小值也发生变化了呢？这是我想不通的一点，难道这种训练方式，提供了一种通往sub minima的捷径？

这种训练方式，按作者的说法有3个好处
> (1) a lower minimum test loss as shown by the yellow and purple curves,
>  (2) faster initial convergence,as shown by the yellow and purple curves, and
>   (3) greater convergence stability over a larger range of learning rates, as shown by the yellow curve.

文章还说，使用decreasing momentum的方法跟使用最佳constant momentum获得的效果相仿，但是允许使用更大的learning rate。

此外他还表明，使用较大的momentum虽然会避免陷入鞍点但是会影响最终的收敛。所以提倡使用相对小的momentum。

在使用constant learning rate的时候，decreasing momentum并没有表现的如best constant momentum一样。此时他建议在0.9～0.99之间搜寻最佳momentum

### Weight Decay
文章表示，weight decay的值跟learning rate和momentum不一样，实验表明这个超参数适合在训练过程中保持为constant，如果不清楚应该设置什么样的大小，那么$10^{-3}, 10^{-4}, 10^{-5}$ 值得你试一下。

此外，复杂的网络结构最好使用较小的weight decay， 简单的网络结构则最好使用较大的weight decay，这里的前提假设是，复杂的网络结构本身就具有一定的正则化功能。好像有点道理。

如果需要搜索最佳weight decay的值，那么使用指数二分搜索会比较好，而不是单纯的数值二分搜索

