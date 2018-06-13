---
layout: post
title: 谈谈Linux中的～
author: Gin 
excerpt_separator: <!--more-->
tag: Linux, Shell
---

使用过Linux的同学都熟悉～这个符号，使用它可以快速访问用户根目录
```bash
cd ~ # 打开用户根目录
```

这个跟环境变量HOME的作用时相同的
```bash
cd $HOME
```
但是仔细区别一下这两者有什么区别呢？
```bash
echo ~
output: /Users/Gin

echo $HOME
output: /Users/Gin
```
貌似没有区别，但是再试试这个
{% highlight bash %}
echo "~"
output: ~

echo "$HOME"
output: /Users/Gin
{% endhighlight %}

开始不一样了

<!--more-->
那么让我们来试试加上引号的后还能不能使用cd
```bash
cd "~"
output: cd: no such file or directory: ~

cd "$HOME"
output: 跳转到～
```
为什么加上引号以后环境变量还是可以正常使用，而～就不行了？

""在linxu中用于修饰字符串，但是如果内部包含有$，那么会优先寻找$后对应的环境变量，并替换成该环境变量的值而如果，该环境变量没有找到就会是空，这是相对特殊的一种处理方式，而～如果被“”修饰了就只是代表“～”这个符号而已

所以如果在shell脚本中要用到用户的根目录作为相对路径，千万不要使用～作为赋值给变量的一部分
```bash
p="~/work/project_1/"
cd $p
output: cd: no such file or directory: ~/work/project_1
```
这种做法是错误的，可以这么使用
```bash
p="$HOME/work/project_1"
cd $p
```
也可以这么使用
```bash
p=~"/work/project_1"
cd $p
```

除此之外，另一个用于修饰字符的符号‘’，则更强力一点，‘$HOME’就表示$HOME这个字符，不受其中任何特殊符号的影响如果想使用