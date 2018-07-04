---
layout: post
title: 浮点运算的Magic
author: Gin 
excerpt_separator: <!--more-->
categories: [Tech]
---
> The algorithm accepts a 32-bit floating-point number as the input and stores a halved value for later use. Then, treating the bits representing the floating-point number as a 32-bit integer, a logical shift right by one bit is performed and the result subtracted from the magic number 0x5F3759DF. This is the first approximation of the inverse square root of the input. Treating the bits again as a floating-point number, it runs one iteration of Newton's method, yielding a more precise approximation.
> 
> It is not known how the constant was originally derived, though investigation has shed some light on possible methods.


迷一样的算法


>The key of the fast inverse square root was to directly compute an approximation by utilizing the structure of floating-point numbers, proving faster than table lookups.

利用了IEEE754浮点数的结构


首先来复习一下IEEE754的浮点数构成


![single precision](https://i.loli.net/2018/06/26/5b32066b4fd99.png
)


单精度，32位的浮点数构成是1位符号位，8位指数位，23位分数位，举例来说0.15625表示成二进制小数就是0.00101


$$0.00101_{2} = 1.01 \times 2^{-3}$$


因为0.00101是正数，符号位为0，分数位从高到低填入小数点后的数位，就是01，指数位需要注意一下，由于指数位有8位，所以其能表示的范围是00000000～11111111，对应十进制的0～255，在计算指数位的值时，需要对当前数的指数加上一个偏置项，这里取127，换言之，原二进制指数最小可以是-127，最大可以是128，但是-127和128是用来表示特殊数字的，因此不可用，故指数的可取范围是-126到127


> If x is a positive [normal number](https://en.wikipedia.org/wiki/Normal_number_\(computing\)):
>
>    $${\displaystyle x=2^{e_{x}}(1+m_{x})}$$
> 
> then there is 
> 
>    $${\displaystyle \log _{2}(x)=e_{x}+\log _{2}(1+m_{x})}$$
> 
> but since mx ∈ [0, 1), the logarithm on the right hand side can be approximated by[[17]](https://en.wikipedia.org/wiki/Fast_inverse_square_root#cite_note-FOOTNOTEMcEniry20073-21)
> 
> $${\displaystyle \log _{2}(1+m_{x})\approx m_{x}+\sigma } \log _{2}(1+m_{x})\approx m_{x}+\sigma $$

> 
> where σ is a free parameter used to tune the approximation. For example, σ = 0 yields exact results at both ends of the interval, while σ ≈ 0.0430357 yields the [optimal approximation](https://en.wikipedia.org/wiki/Approximation_theory#Optimal_polynomials) (the best in the sense of the [uniform norm](https://en.wikipedia.org/wiki/Uniform_norm) of the error).
> 
> The integer aliased to a floating point number (in blue), compared to a scaled and shifted logarithm (in gray). 
> 
> Thus there is the approximation 
> 
>    $${\displaystyle \log _{2}(x)\approx e_{x}+m_{x}+\sigma .}$$
> 
> Alternately, interpreting the bit-pattern of x as an integer yields[[note 5]](https://en.wikipedia.org/wiki/Fast_inverse_square_root#cite_note-22)
> 
>    
>    $${\displaystyle {\begin{aligned}I_{x}&=E_{x}L+M_{x}\\&=L(e_{x}+B+m_{x})\\&=L(e_{x}+m_{x}+\sigma +B-\sigma )\\&\approx L\log _{2}(x)+L(B-\sigma ).\end{aligned}}}$$    


> It then appears that Ix is a scaled and shifted piecewise-linear approximation of log2(x), as illustrated in the figure on the right. In other words, log2(x) is approximated by
> 
>    $${\displaystyle \log _{2}(x)\approx {\frac {I_{x}}{L}}-(B-\sigma ).}$$
> 
> ### First approximation of the result
> 
> 
> The calculation of $$y = 1/ \sqrt{x}$$ is based on the identity 
> 
>    $${\displaystyle \log _{2}(y)=-{\frac {1}{2}}\log _{2}(x)}$$
> 
> Using the approximation of the logarithm above, applied to both x and y, the above equation gives: 
> 
>    $${\displaystyle {\frac {I_{y}}{L}}-(B-\sigma )\approx -{\frac {1}{2}}{\biggl (}{\frac {I_{x}}{L}}-(B-\sigma ){\biggr )}}$$
> 
> Thus, an approximation of Iy is: 
> 
>    $${\displaystyle I_{y}\approx {\frac {3}{2}}L(B-\sigma )-{\frac {1}{2}}I_{x}}$$
> 
> which is written in the code as 
> 
> ```c
> i = 0x5f3759df - ( i >> 1 );
> ```
> 
> The first term above is the magic number
> 
>    $${\displaystyle {\frac {3}{2}}L(B-\sigma )={\text{0x5F3759DF}}}$$
> 
> from which it can be inferred σ ≈ 0.0450466. The second term, ½ Ix, is calculated by shifting the bits of Ix one position to the right.[[18]](https://en.wikipedia.org/wiki/Fast_inverse_square_root#cite_note-FOOTNOTEHennesseyPatterson1998305-23)

>{:class="center"}
> ![Figure1](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Log_by_aliasing_to_int.svg/640px-Log_by_aliasing_to_int.svg.png)
> 
> {:class="image-caption"}
> *The integer aliased to a floating point number (in blue), compared to a scaled and shifted logarithm (in gray).*


[Further Reading, WiKi:Single-precision_floating-point_format](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)


Newton’s method we’ll talk later.
