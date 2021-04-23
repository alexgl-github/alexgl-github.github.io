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


### For input vector X of size M, and dense layer with $$ M $$ inputs and $$ N $$ outputs, output Y is:

$$ Y = X * W $$

where X is input vector:

$$ X = \left( \begin{array}{ccc}
x_{0} & x_{1} & \ldots & x_{M-1} \\
\end{array} \right)
$$

$$ W $$ is weights matrix:

$$
W = \left( \begin{array}{ccc}
w_{0,0} & w_{0,1} & \ldots & w_{0,N-1} \\
w_{1,0} & w_{1,1} & \ldots & w_{1,N-1} \\
\vdots & \vdots & \ldots & \vdots \\
w_{M-1,0} & w_{M-1,1} & \ldots & w_{M-1,N-1} \\
\end{array} \right)
$$

$$ Y $$ is output vector:

$$ Y = \left( \begin{array}{ccc}
y_{0} & y_{1} & \ldots & y_{n-1} \\
\end{array} \right)
$$

$$ \hat Y $$ is expected output vector:

$$ \hat Y = \left( \begin{array}{ccc}
\hat y_{0} & \hat y_{1} & \ldots & \hat y_{n-1} \\
\end{array} \right)
$$

Mean Squared Error loss between predicted $$ Y $$ and expected $$ \hat Y $$ is

$$ E = \frac {1} {N} \sum_{i=0}^{N-1} ( \hat y_{i} - y_{i}) $$



