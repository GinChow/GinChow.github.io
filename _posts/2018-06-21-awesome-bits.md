---
layout: post
title: awesome-bits
author: Gin 
excerpt_separator: <!--more-->
categories: [Tech]
---

[Original Reference Link](https://github.com/keon/awesome-bits)

# awesome-bits [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of awesome bitwise operations and tricks
>
> Maintainer - [Keon Kim](https://github.com/keonkim)
> Please feel free to [pull requests](https://github.com/keonkim/awesome-bits/pulls)




## Integers

```c
x | (1<<n)
```

 ```c
 x & ~(1<<n)
 ```

```c
x ^ (1<<n)
```
**Round up to the next power of two**

```c
unsigned int v; //only works if v is 32 bit
v--;
v |= v >> 1;
v |= v >> 2;
v |= v >> 4;
v |= v >> 8;
v |= v >> 16;
v++;
```
**Get the maximum integer**

```c
int maxInt = ~(1 << 31);
int maxInt = (1 << 31) - 1;
int maxInt = (1 << -1) - 1;
int maxInt = -1u >> 1;
```
**Get the minimum integer**

```c
int minInt = 1 << 31;
int minInt = 1 << -1;
```
**Get the maximum long**

```c
long maxLong = ((long)1 << 127) - 1;
```
**Multiply by 2**

```c
n << 1; // n*2
```
**Divide by 2**

```c
n >> 1; // n/2
```
**Multiply by the m power of 2**

```c
n << m;
```
**Divide by the mth power of 2**

```c
n >> m;
```
**Check Equality**


```c
(a^b) == 0; // a == b
!(a^b) // use in an if
```
**Check if a number is odd**

```c
(n & 1) == 1;
```
**Exchange (swap) two values**

```c
//version 1
a ^= b;
b ^= a;
a ^= b;

//version 2
a = a ^ b ^ (b = a)
```
**Get the absolute value**

```c
//version 1
x < 0 ? -x : x;

//version 2
(x ^ (x >> 31)) - (x >> 31);
```
**Get the max of two values**

```c
b & ((a-b) >> 31) | a & (~(a-b) >> 31);
```
**Get the min of two values**

```c
a & ((a-b) >> 31) | b & (~(a-b) >> 31);
```
**Check whether both numbers have the same sign**

```c
(x ^ y) >= 0;
```
**Flip the sign**

```c
i = ~i + 1; // or
i = (i ^ -1) + 1; // i = -i
```
**Calculate**![formula](http://latex.codecogs.com/svg.latex?2%5E%7Bn%7D "2^{n}")

```c
1 << n;
```
**Whether a number is power of 2**

```c
n > 0 && (n & (n - 1)) == 0;
```
**Modulo 2^n against m**

```c
m & ((1 << n) - 1);
```
**Get the average**

```c
(x + y) >> 1;
((x ^ y) >> 1) + (x & y);
```
**Get the mth bit of n (from low to high)**

```c
(n >> (m-1)) & 1;
```
**Set the mth bit of n to 0 (from low to high)**

```c
n & ~(1 << (m-1));
```
**Check if nth bit is set**

```c
if (x & (1<<n)) {
  n-th bit is set
} else {
  n-th bit is not set
}
```
**Isolate (extract) the right-most 1 bit**

```c
x & (-x)
```
**Isolate (extract) the right-most 0 bit**

```c
~x & (x+1)
```

**Set the right-most 0 bit to 1**

```c
x | (x+1)
```

**Set the right-most 1 bit to 0**

```c
x & (x-1)
```

**n + 1**
```c
-~n
```
**n - 1**

```c
~-n
```
**Get the negative value of a number**

```c
~n + 1;
(n ^ -1) + 1;
```
**`if (x == a) x = b; if (x == b) x = a;`**

```c
x = a ^ b ^ x;
```
**Swap Adjacent bits**

```c
((n & 10101010) >> 1) | ((n & 01010101) << 1)
```
**Different rightmost bit of numbers m & n**

```c
(n^m)&-(n^m) // returns 2^x where x is the position of the different bit (0 based)
```
**Common rightmost bit of numbers m & n**

```c
~(n^m)&(n^m)+1 // returns 2^x where x is the position of the common bit (0 based)
```
## Floats

These are techniques inspired by the [fast inverse square root method.](https://en.wikipedia.org/wiki/Fast_inverse_square_root) Most of these
are original.

**Turn a float into a bit-array (unsigned uint32_t)**

```c
#include <stdint.h>
typedef union {float flt; uint32_t bits} lens_t;
uint32_t f2i(float x) {
  return ((lens_t) {.flt = x}).bits;
}
```
*Caveat: Type pruning via unions is undefined in C++; use `std::memcpy` instead.*

**Turn a bit-array back into a float**

```c
float i2f(uint32_t x) {
  return ((lens_t) {.bits = x}).flt;
}
```

**Approximate the bit-array of a *positive* float using `frexp`**

*`frexp` gives the 2^n decomposition of a number, so that `man, exp = frexp(x)` means that man * 2^exp = x and 0.5 <= man < 1.*

```c
man, exp = frexp(x);
return (uint32_t)((2 * man + exp + 125) * 0x800000);
```
*Caveat: This will have at most 2^-16 relative error, since man + 125 clobbers the last 8 bits, saving the first 16 bits of your mantissa.*

**Fast Inverse Square Root**

```c
return i2f(0x5f3759df - f2i(x) / 2);
```
*Caveat: We're using the `i2f` and the `f2i` functions from above instead.*

See [this Wikipedia article](https://en.wikipedia.org/wiki/Fast_inverse_square_root#A_worked_example) for reference.

**Fast nth Root of positive numbers via Infinite Series**

```c
float root(float x, int n) {
#DEFINE MAN_MASK 0x7fffff
#DEFINE EXP_MASK 0x7f800000
#DEFINE EXP_BIAS 0x3f800000
  uint32_t bits = f2i(x);
  uint32_t man = bits & MAN_MASK;
  uint32_t exp = (bits & EXP_MASK) - EXP_BIAS;
  return i2f((man + man / n) | ((EXP_BIAS + exp / n) & EXP_MASK));
}
```

See [this blog post](http://www.phailed.me/2012/08/somewhat-fast-square-root/) regarding the derivation.

**Fast Arbitrary Power**

```c
return i2f((1 - exp) * (0x3f800000 - 0x5c416) + f2i(x) * exp)
```

*Caveat: The `0x5c416` bias is given to center the method. If you plug in exp = -0.5, this gives the `0x5f3759df` magic constant of the fast inverse root method.*

See [these set of slides](http://www.bullshitmath.lol/FastRoot.slides.html) for a derivation of this method.

**Fast Geometric Mean**

The geometric mean of a set of `n` numbers is the nth root of their
product.

```c
#include <stddef.h>
float geometric_mean(float* list, size_t length) {
  // Effectively, find the average of map(f2i, list)
  uint32_t accumulator = 0;
  for (size_t i = 0; i < length; i++) {
    accumulator += f2i(list[i]);
  }
  return i2f(accumulator / n);
}
```
See [here](https://github.com/leegao/float-hacks#geometric-mean-1) for its derivation.

**Fast Natural Logarithm**

```c
#DEFINE EPSILON 1.1920928955078125e-07
#DEFINE LOG2 0.6931471805599453
return (f2i(x) - (0x3f800000 - 0x66774)) * EPSILON * LOG2
```

*Caveat: The bias term of `0x66774` is meant to center the method. We multiply by `ln(2)` at the end because the rest of the method computes the `log2(x)` function.*

See [here](https://github.com/leegao/float-hacks#log-1) for its derivation.

**Fast Natural Exp**

```c
return i2f(0x3f800000 + (uint32_t)(x * (0x800000 + 0x38aa22)))
```

*Caveat: The bias term of `0x38aa22` here corresponds to a multiplicative scaling of the base. In particular, it
corresponds to `z` such that 2z = e*

See [here](https://github.com/leegao/float-hacks#exp-1) for its derivation.

## Strings

**Convert letter to lowercase:**

```c
OR by space => (x | ' ')
Result is always lowercase even if letter is already lowercase
eg. ('a' | ' ') => 'a' ; ('A' | ' ') => 'a'
```

**Convert letter to uppercase:**

```c
AND by underline => (x & '_')
Result is always uppercase even if letter is already uppercase
eg. ('a' & '_') => 'A' ; ('A' & '_') => 'A'
```
**Invert letter's case:**

```c
XOR by space => (x ^ ' ')
eg. ('a' ^ ' ') => 'A' ; ('A' ^ ' ') => 'a'
```
**Letter's position in alphabet:**

```c
AND by chr(31)/binary('11111')/(hex('1F') => (x & "\x1F")
Result is in 1..26 range, letter case is not important
eg. ('a' & "\x1F") => 1 ; ('B' & "\x1F") => 2
```
**Get letter's position in alphabet (for Uppercase letters only):**

```c
AND by ? => (x & '?') or XOR by @ => (x ^ '@')
eg. ('C' & '?') => 3 ; ('Z' ^ '@') => 26
```
**Get letter's position in alphabet (for lowercase letters only):**

```c
XOR by backtick/chr(96)/binary('1100000')/hex('60') => (x ^ '`')
eg. ('d' ^ '`') => 4 ; ('x' ^ '`') => 24
```

## Miscellaneous

**Fast color conversion from R5G5B5 to R8G8B8 pixel format using shifts**

```c
R8 = (R5 << 3) | (R5 >> 2)
G8 = (R5 << 3) | (R5 >> 2)
B8 = (R5 << 3) | (R5 >> 2)
```
Note: using anything other than the English letters will produce garbage results

## Additional Resources

* [Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html)
* [Floating Point Hacks](https://github.com/leegao/float-hacks)
* [Hacker's Delight](http://www.hackersdelight.org/)
* [The Bit Twiddler](http://bits.stephan-brumme.com/)


