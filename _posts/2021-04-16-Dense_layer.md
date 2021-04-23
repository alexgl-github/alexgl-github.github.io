---
layout: post
mathjax: true
title:  "Dense layer with backpropagation in C++"
date:   2021-04-16 23:41:54 +0000
categories: github jekyll
---

## Dense layer with backpropagation in C++

Let's code a simple one layer DNN in C++.
The network should support forward and back propagation.
Initially we'll try to keep it as simple as possible.
We assume:
- there's no bias in the dense layer;
- there's no non-linear activation;
- loss function is Mean Squared Error.


### For input vector X of size M, and dense layer with M inputs and N outputs, output will be:

$$ Y = X * W

X is input vector:

$$ X = \left( \begin{array}{ccc}
x_{00} & x_{01} & \ldots & x_{0M} \\
\end{array} \right)
$$


W is weights matrix:

$$
W = \left( \begin{array}{ccc}
w_{00} & w_{01} & \ldots & w_{0N} \\
w_{10} & w_{11} & \ldots & w_{0N} \\
\vdots & \vdots & \ldots & w_{0N} \\
w_{M0} & w_{M1} & \ldots & w_{0N} \\
\end{array} \right)
$$

$$ r = h = \sqrt{\frac {1} {2}} = \sqrt{\frac {N} {N+1}} \sqrt{\frac {N+1} {2N}} $$

