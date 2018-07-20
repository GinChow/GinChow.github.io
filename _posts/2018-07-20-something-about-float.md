---
layout: post
title: Python中的浮点数
author: Gin 
excerpt_separator: <!--more-->
tag: Python, Shell
categories: [Python,Tech]
---


最近遇到python中的float64转float32精度失真的问题，

比如：

```python
>>> f1 = np.float64(3.13415126453231423)
>>> f1
3.1341512645323144
>>> f2 = f1.astype(np.float32)
>>> f2
3.1341512
```

可以看到float64转换成float32并没有遵循我们常规的四舍五入法则, 2后面的6没有进位，这是为什么呢？
<!--more-->

首先让我们回到IEEE 754标准
[WiKi IEEE 754 standard](https://zh.wikipedia.org/wiki/IEEE_754)

小数部分的数字其实都是2的负幂次的和

我们将这两个浮点数的bytes code打印出来，为了便于观看，我们使用小端模式，也就是高位高地址，低位低地址(大端模式就是高位低地址，低位高地址)详参
[Endian Mode](https://zh.wikipedia.org/wiki/%E5%AD%97%E8%8A%82%E5%BA%8F)


![Big Endian](https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Big-Endian.svg/560px-Big-Endian.svg.png)
![Little Endian](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Little-Endian.svg/560px-Little-Endian.svg.png)


|小端模式 |←| |地址增长方向 ||
| :----: | :--------: | :---: | :----|
| 0x0A | 0x0B | 0x0C | 0x0D |



|大端模式| → |  |地址增长方向||
|:--:|:---:|:---:|:---|
|0x0D|0x0C|0x0B|0x0A|



```python
import struct
>>> f1_b = struct.pack('>d', f1)
>>> f1_b
b'@\t\x12\xbd\xe7F\xaa\x00'
>>> f2_b = struct.pack('>f', f2)
>>> f2_b
b'@H\x95\xef'
```

struct.pack()的作用是获取对象的字节码，‘>’表示我们需要的小端模式
可以看到我们得到的对象的确是一堆16进制的编码，等等，为什么会乱入一下莫名其妙的符号，‘@’‘H’‘\t’是什么鬼
不要慌张, 让我们继续探究

```python
>>> hex(ord('@'))
'0x40'
>>> hex(ord('\t'))
'0x9'
>>> hex(ord('H'))
'0x48'
```

look其实这些字符的表示也是16进制的，但是在显示的时候被当成ascii码中定义的字符了

```python
>>> c = b'\x40\x09\x12\xbd\xe7\x46\xaa\x00'
>>> c
b'@\t\x12\xbd\xe7F\xaa\x00'
>>> d = b'\x40\x48\x95\xef'
>>> d
b'@H\x95\xef'
```

可以看到，两者是等价的。

好了，回到为什么没有四舍五入的问题，默认的结果是

```python
>>> f2
3.1341512
```

它的bytes code是

```python
>>> f2_b
b'@H\x95\xef'
```

假如我们在分数上的最小位加上1，bytes code就会变成
```python
b'@H\x95\xf0'
```

我们使用struct.unpack()来查看其表示的浮点数

```python
>>> res = struct.unpack('>f', b'@H\x95\xf0')
>>> res
(3.1341514587402344,)
>>> res = struct.unpack('>f', b'@H\x95\xef')
>>> res
(3.1341512203216553,)
>>> f1
3.1341512645323144
```

发现了什么问题？ 我们对最小精度加了1，但是最后转化得到的decimal,但是相较于原始的float64数字其小数点后第7位差别就很大了，从2一下变成了4，显然这是不可接受的

计算机内部的计算都是二进制形式的，所以从数学角度来看有些结果是非常反直觉的，但是毕竟不能要求太多，工具本身的固有局限性是无法避免的

