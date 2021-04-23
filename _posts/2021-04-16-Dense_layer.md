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


### First, let's implement Python implementation with TF2/Keras. We''l use to validate C++ code.


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


### After running Python example we get:

{% highlight bash %}

$ python3 dense.py
input x=[2.  0.5 1. ]
output y=[3.5 3.5]
expected output y_true=[1.5 1. ]
loss=5.125
dloss_dy=[2.  2.5]
grad=
[[4.   5.  ]
 [1.   1.25]
 [2.   2.5 ]]
vars=
[[-3.   -4.  ]
 [ 0.   -0.25]
 [-1.   -1.5 ]]

{% endhighlight %}


Let's code the same example in C++

{% highlight python %}
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <functional>
#include <array>
#include <iterator>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

/*
 * Constant 1.0 weight intializer
 */
static auto ones_initializer = []() -> float
{
  return 1.0;
};

/*
 * Dense layer class template
 *
 * Parameters:
 *  num_inputs: number of inputs to Dense layer
 *  num_outputs: number of Dense layer outputs
 *  T: input, output, and weights type in the dense layer
 *  initializer: weights initializer function
 */
template<size_t num_inputs, size_t num_outputs, typename T = float, T (*initializer)()=ones_initializer>
struct Dense
{

  typedef array<T, num_inputs> input_vector;
  typedef array<T, num_outputs> output_vector;

  vector<input_vector> weights;

  /*
   * Dense layer constructor
   */
  Dense()
  {
    /*
     * Create num_outputs x num_inputs matric
     */
    weights.resize(num_outputs);
    for (input_vector& w: weights)
      {
        generate(w.begin(), w.end(), *initializer);
      }
  }

  /*
   * Dense layer forward pass
   */
  array<T, num_outputs> forward(const input_vector& x)
  {
    /*
     * Check for input size mismatch
     */
    assert(x.size() == weights[0].size());

    /*
     * Layer output is dot product of input with weights
     */
    array<T, num_outputs> activation;
    int idx = 0;
    for (auto w: weights)
      {
        T val = inner_product(x.begin(), x.end(), w.begin(), 0.0);
        activation[idx++] = val;
      }

    return activation;
  }

  /*
   * Dnse layer backward pass
   */
  void backward(input_vector& input, output_vector dloss_dy)
  {
    /*
     * Weight update according to SGD algorithm with momentum = 0.0 is:
     *  w = w - learning_rate * d_loss/dw
     *
     * For simplicity assume learning_rate = 1.0
     *
     * d_loss/dw = dloss/dy * dy/dw
     *
     * dy/dw is :
     *  y = w[0]*x[0] + w[1] * x[1] +... + w[n] * x[n]
     *  dy/dw[i] = x[i]
     *
     * For clarity we:
     *  assume learning_rate = 1.0
     *  first compute dw
     *  second update weights by subtracting dw
     */

    /*
     * compute dw
     * dw = outer(x, de_dy)
     */
    vector<input_vector> dw;
    for (auto dloss_dyi: dloss_dy)
      {
        auto row = input;
        for_each(row.begin(), row.end(), [dloss_dyi](T &xi){ xi *= dloss_dyi;});
        dw.push_back(row);
      }

    /*
     * compute w = w - dw
     * assume learning rate = 1.0
     */
    transform(weights.begin(), weights.end(), dw.begin(), weights.begin(),
              [](input_vector& left, input_vector& right)
              {
                transform(left.begin(), left.end(), right.begin(), left.begin(), minus<T>());
                return left;
              }
              );
  }

  /*
   * Helper function to convert Dense layer to string
   * Used for printing the layer
   */
  operator std::string() const
  {
    string ret;

    for (int y=0; y < weights[0].size(); y++)
      {
        for (int x=0; x < weights.size(); x++)
          {
            ret += to_string(weights[x][y]) + " ";
          }
        ret += "\n";
      }
    return ret;
  }

  /*
   * Helper function to cout Dense layer object
   */
  friend ostream& operator<<(ostream& os, const Dense& dense)
  {
    os << (string)dense;
    return os;
  }

};

/*
 * Mean Squared Error loss class
 * Parameters:
 *  num_inputs: number of inputs to MSE function.
 *  T: input type, float by defaut.
 */
template<size_t num_inputs, typename T = float>
struct MSE
{
  /*
   * Forward pass computes MSE loss for inputs yhat (label) and y (predicted)
   */
  static T forward(array<T, num_inputs> yhat, array<T, num_inputs> y)
  {
    T loss = transform_reduce(yhat.begin(), yhat.end(), y.begin(), 0.0, plus<T>(),
                              [](T& left, T& right)
                              {
                                return (left - right) * (left - right);
                              }
                              );
    return loss / num_inputs;
  }

  /*
   * Backward pass computes dloss/dy for inputs yhat (label) and y (predicted)
   *
   * loss = sum((yhat[i] - y[i])^2) / N
   *   i=0...N-1
   *   where N is number of inputs
   *
   * d_loss/dy[i] = 2 * (yhat[i] - y[i]) * (-1) / N
   * d_loss/dy[i] = 2 * (y[i] - yhat[i]) / N
   *
   */
  static array<T, num_inputs> backward(array<T, num_inputs> yhat, array<T, num_inputs> y)
  {
    array<T, num_inputs> de_dy;

    transform(yhat.begin(), yhat.end(), y.begin(), de_dy.begin(),
              [](T& left, T& right)
              {
                return 2 * (right - left) / num_inputs;
              }
              );
    return de_dy;
  }

};


int main(void)
{
  const int num_inputs = 3;
  const int num_outputs = 2;
  const int num_iterations = 1000;
  auto print_fn = [](const float& x)  -> void {printf("%.5f ", x);};

  array<float, num_inputs> x = {2.0, 0.5, 1.0};
  array<float, num_outputs> y_true = {1.5, 1.0};

  /*
   * Create dense layer and MSE loss
   */
  Dense<num_inputs, num_outputs> dense;
  MSE<num_outputs> mse_loss;

  /*
   * Compute Dense layer output y for input x
   */
  auto y = dense.forward(x);

  /*
   * Copute MSE loss for output y and expected y_true
   */
  auto loss = mse_loss.forward(y_true, y);

  /*
   * Run inference 1000 times and benchmark dense layer latency
   */
  auto ts = high_resolution_clock::now();
  for (auto iter = 0;  iter < num_iterations; iter++) [[likely]] dense.forward(x);
  auto te = high_resolution_clock::now();
  auto dt_us = (float)duration_cast<microseconds>(te - ts).count() / num_iterations;

  /*
   * Print DNN input x
   */
  printf("input x=");
  for_each(x.begin(), x.end(), print_fn);
  printf("\n");

  /*
   * Print DNN output y
   */
  printf("outut y=");
  for_each(y.begin(), y.end(), print_fn);
  printf("\n");

  /*
   * Print expected output y_true
   */
  printf("expected outut y=");
  for_each(y_true.begin(), y_true.end(), print_fn);
  printf("\n");

  /*
   * Print loss for output y and label y_true
   */
  printf("loss: %f\n", loss);

  /*
   * Compute dloss/dy gradients
   */
  auto dloss_dy = mse_loss.backward(y_true, y);

  /*
   * Back propagate loss
   */
  dense.backward(x, dloss_dy);

  /*
   * print dloss/dy
   */
  printf("loss gradient: ");
  for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
  printf("\n");

  /*
   * Print updated Dense layer weights
   */
  printf("updated dense layer weights:\n%s", ((string)dense).c_str());

  /*
   * Print average latency
   */
  printf("time dt=%f usec\n", dt_us);

return 0;
}

{% endhighlight %}


### After compiling and running C++ example we get:

{% highlight bash %}

$ g++ -o dense -std=c++2a dense.cpp && ./dense
input x=2.00000 0.50000 1.00000
outut y=3.50000 3.50000
expected outut y=1.50000 1.00000
loss: 5.125000
loss gradient: 2.00000 2.50000
updated dense layer weights:
-3.000000 -4.000000
0.000000 -0.250000
-1.000000 -1.500000
time dt=0.113000 usec

{% endhighlight %}


Python source code for this example is at [python_source_code] [/src/dense.py]
C++ implementation is at [cpp_source_code] [/src/dense.cpp]

[python_source_code]:  {{ site.url }}/src/dense.py
[cpp_source_code]:  {{ site.url }}/src/dense.cpp
