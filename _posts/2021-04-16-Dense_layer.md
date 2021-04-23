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


### Forward path for input vector X of size M, and dense layer with $$ M $$ inputs and $$ N $$ outputs, output Y is:

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
y_{0} & y_{1} & \ldots & y_{N-1} \\
\end{array} \right)
$$

$$ \hat Y $$ is expected output vector:

$$ \hat Y = \left( \begin{array}{ccc}
\hat y_{0} & \hat y_{1} & \ldots & \hat y_{N-1} \\
\end{array} \right)
$$

Mean Squared Error loss between predicted $$ Y $$ and expected $$ \hat Y $$ is

$$ E = \frac {1} {N} \sum_{i=0}^{N-1} ( \hat y_{i} - y_{i})^2 $$


First, let's implement Python implementation with TF2/Keras, which we''l use to validate C++ code


{% highlight python %}

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

num_inputs = 3
num_outputs = 2

# Create one layer model
model = tf.keras.Sequential()

# No bias, no activation, initialize weights with 1.0
model.add(Dense(units=num_outputs, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.ones()))

# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Arbitrary model input
x = np.array([2.0, 0.5, 1])

# Expected output
y_true = np.array([1.5, 1.0])


# SGD update rule for parameter w with gradient g when momentum is 0 is as follows:
#   w = w - learning_rate * g
#
#   For simplicity make learning_rate=1.0
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0)

# Get model output y for input x, compute loss, and record gradients
with tf.GradientTape(persistent=True) as tape:

    # get model output y for input x
    # add newaxis for batch size of 1
    xt = tf.convert_to_tensor(x[np.newaxis, ...])
    tape.watch(xt)
    y = model(xt)

    # obtain MSE loss
    loss = loss_fn(y_true, y)

    # loss gradient with respect to loss input y
    dloss_dy = tape.gradient(loss, y)

    # adjust Dense layer weights
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

# print model input and output excluding batch dimention
print(f"input x={x}")
print(f"output y={y[0]}")
print(f"expected output y_true={y_true}")

# print MSE loss
print(f"loss={loss}")

# print loss gradients
print(f"dloss_dy={dloss_dy[0].numpy()}")

# print weight gradients d_loss/d_w
print(f"grad=\n{grad[0].numpy()}")

# print updated dense layer weights
print(f"vars=\n{model.trainable_variables[0].numpy()}")

{% endhighlight %}




