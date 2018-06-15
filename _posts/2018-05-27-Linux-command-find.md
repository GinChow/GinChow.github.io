---
layout: post
title: Linux command find
author: Gin 
excerpt_separator: <!--more-->
categories: [Linux, Personal]
---

&emsp;&emsp;进新公司两个星期了，前两天在处理数据的时候遇到了个问题，公司有个tools team，这帮家伙写的工具文档基本等于没有，
help信息也很少，实际操作起来只能靠自己摸索。我在使用他们给的一个工具时，发现这个工具生成的文件，文件名必须带有指定前缀，还不能省略！
```bash
cmd -g -a "abc" #-a不能省略...
```
<!--more-->
然后，他们提供的另一个工具，则必须指定两个文件夹，每个文件夹中的文件名必须相同才会一一匹配上。你好歹要让加个前缀，好跟上一个工具匹配啊。
额，这真的是个奇葩的team。Fine，现在要做的就是，把生成文件的多余前缀去掉。看起来是一个挺轻松的活，用python写个脚本分分钟解决。
<br  />&emsp;&emsp;但是!!!
<br  />&emsp;&emsp;身为一个优雅的One-liner Coder怎么能为了做这种简单的事而专门写个脚本呢？这太不优雅了。
我决定使用Shell命令来做。<br/>
乍看似乎挺简单的，通配符*是我们的好兄弟。
```bash
ls prefix*.ext | mv prefix*.ext *.ext # 
```
这条命令看起来不错， 但实际上shell并不能执行它，因为管道命令并不知道*匹配的是什么东西。<br/>
也许我还是应该写一个shell脚本？
```bash
#!bin/bash
for file_name in $(ls /dir/prefix*.ext);
do
	new_file_name = "${file_name##"prefix"}"
	mv file_name new_file_name
done
```
这里使用了另一个操作符#，作用是从左往右过滤字符串中的子串，单个#是非贪婪模式，##则是贪婪模式，相对的还有一个%操作符，是从右往左过滤字符串子串。
这算是点字符串操作小技巧。<br/>
好了，现在这个脚本已经能够满足工作需求，但实际上我在运行这个脚本的时候发现异常的慢，文件夹下大概有近两万个文件，这个脚本需要运行近20分钟才能将
所有的文件名去除前缀。有点不满意啊。。。。<br/>
Google了一下，发现stackoverflow上面有人提过类似的问题，但还是有一点区别，他的要求是在文件夹递归的进行这样的操作，如果照上面这个脚本的思路来写，并不是
很方便。