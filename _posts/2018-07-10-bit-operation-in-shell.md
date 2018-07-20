---
layout: post
title: Shell中的位运算
author: Gin 
excerpt_separator: <!--more-->
categories: [Tech, Shell]
---

Shell中的位运算，有特殊的使用方式
<!--more-->

```shell
b=1
b=$b<<1
# 这种写法是不行的会与重定向输入冲突
```


```shell
b=1
b=$((b<<1))
# 这种写法可以，执行后b的值变成2
```

其他位运算符的操作，也可以用这种形式书写

[相关问题链接](https://stackoverflow.com/questions/40667382/how-to-perform-bitwise-operations-on-hexadecimal-numbers-in-bash)


