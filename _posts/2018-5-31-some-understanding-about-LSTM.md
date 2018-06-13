---
layout: post
title: LSTM的一点理解
author: Gin 
excerpt_separator: <!--more-->
---

这个礼拜开始读LSTM相关的论文，一直以来，LSTM是与ResNet shortcut连接齐名的一种解决Gradient Vanishing的方法，事实上有的文章指出
LSTM中cell state之间的连接，就是某种程度上的short cut，即通过加法门将top层的gradient向bottom层传递，这听上去十分合理。但是ResNet
的short cut连接