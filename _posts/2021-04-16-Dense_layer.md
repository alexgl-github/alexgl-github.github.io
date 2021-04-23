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


### For input vector X of size M, and dense layer with $$ M $$ inputs and $$ N $$ outputs, output Y will be:

$$ Y = X * W $$

X is input vector:

$$ X = \left( \begin{array}{ccc}
x_{0} & x_{1} & \ldots & x_{M-1} \\
\end{array} \right)
$$

$$ W $$ is weights matrix:

$$
W = \left( \begin{array}{ccc}
w_{00} & w_{01} & \ldots & w_{0N} \\
w_{10} & w_{11} & \ldots & w_{0N} \\
\vdots & \vdots & \ldots & w_{0N} \\
w_{M0} & w_{M1} & \ldots & w_{0N} \\
\end{array} \right)
$$

$$ Y $$ is output vector:

$$ Y = \left( \begin{array}{ccc}
y_{0} & y_{1} & \ldots & y_{N-1} \\
\end{array} \right)
$$

$$ \hat Y $$ is expected output

Mean Squred Error loss between predicted $$ Y $$ and expected $$ Y_t $$ is

$$ E = frac {1} {N} \sum_{i=0}^{N-1} ({ \hat y_{i} - y_{i}) $$


