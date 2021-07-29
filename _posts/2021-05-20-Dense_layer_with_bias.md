---
layout: post
mathjax: true
title:  "DNN with backpropagation in C++, part 3"
date:   2021-05-20 00:00:00 +0000
categories: github jekyll
---

### Dense layer with backpropagation and bias in C++

In this post I'll modify the previous example adding bias to the DNN dense layers.

Previous example can be found at ["Dense layer with backpropagation in C++, part 2"] [previous_post]

Assumptions are:
- there's no non-linear activation;
- loss function is Mean Squared Error.


### Forward path for input vector X of size M, and 2 layer neural net with $$M$$ inputs and $$N$$ outputs

Two layer neural net output $$\hat Y$$ is:

$$ Y_{1} = X * W_{1} + B_{1}$$

$$ \hat Y = Y_{1} * W_{2} + B_{2}$$

where

$$X$$ is input vector

$$W_{1}$$ are weights for dense layer 1

$$B_{1}$$ is bias vector for dense layer 1

$$Y_{1}$$ is output of dense layer 1

$$W_{2}$$ are weights for dense layer 2

$$B_{2}$$ is bias vector for dense layer 2

$$ \hat Y $$ is predicted output vector:

$$ \hat Y = \left( \begin{array}{ccc}
\hat y_{0} & \hat y_{1} & \ldots & \hat y_{N-1} \\
\end{array} \right)
$$

![mnist_image]({{ site.url }}/images/dotprod_bias.png)

Mean Squared Error (MSE) loss between predicted $$ Y $$ and expected $$ \hat Y $$

Los function has not changed from the previos example.

$$ E = MSE(Y, \hat Y) = \frac {1} {N} \sum_{i=0}^{N-1} ( Y_{i} - \hat Y_{i} )^2 $$

Where $$Y$$ is expected output of neural net:

$$ Y = \left( \begin{array}{ccc}
y_{0} & y_{1} & \ldots & y_{N-1} \\
\end{array} \right)
$$


### Error backpropagation.

For input $$X$$, we want to minimize the MSE difference between out network output and expected output,
by adjusting weights abd biases of both dense layers:

$$\frac {\partial E} {\partial W_{1}}$$, $$\frac {\partial E} {\partial B_{1}}$$

$$\frac {\partial E} {\partial W_{2}}$$, $$\frac {\partial E} {\partial B_{2}}$$


Weights adjustment for both layers is the same as in the previous post.

### Let's find bias adjustment of dense layer 2.

$$ \hat Y = Y_{1} * W_{2} + B_{2} $$

Using chain rule

$$ \frac {\partial E} {\partial B_{2}} =  \frac {\partial E} {\partial \hat Y} * \frac {\partial \hat Y} {\partial B_{2}} $$


where

$$
\frac {\partial E} {\partial \hat Y} = \frac {2 * ( \hat {Y} - Y )} {N}
$$

$$
\frac {\partial \hat Y_{1}} {\partial B_{2}} = 1
$$

Finally, layer 2 bias update is

$$ \frac {\partial E} {\partial B_{2}} = \frac {2 * ( \hat {Y} -  Y )} {N} $$


### Bias adjustment for the weights of dense layer 1.

$$ \hat Y = Y_{1} * W_{2} + B_{2} $$

$$ Y_{1} = X * W_{1} + B_{1} $$

$$ \frac {\partial E} {\partial B_{1}} = \frac {\partial E} {\partial \hat Y} * \frac {\partial \hat Y} {\partial B_{1}} $$

$$ \frac {\partial E} {\partial B_{1}} = \frac {\partial E} {\partial \hat Y} * \frac {\partial \hat Y} {\partial Y_{1}} * \frac {\partial Y_{1}} {\partial B_{1}} $$

where

$$
\frac {\partial E} {\partial \hat Y} = \frac {2 * ( \hat {Y} - Y )} {N}
$$

$$
\frac {\partial \hat Y} {\partial Y_{1}} = W_{2}^T
$$

$$
\frac {\partial Y_{1}} {\partial B_{1}} = 1
$$

Finally, layer 1 bias update is

$$ \frac {\partial E} {\partial B_{1}} = \frac {2 * ( \hat {Y} - Y )} {N} * W_{2}^T $$



### First, let's modify python code from previous example to use bias in dense layers.
As usual we'll use it to validate C++ code in the consecutive section.

For this experiment I've used the following software versions:

{% highlight bash %}
$ python3 -m pip freeze | grep "numpy\|tensorflow"
numpy==1.19.5
tensorflow==2.5.0rc2

$ g++ --version
g++ 9.3.0

{% endhighlight %}


Python code will be very similar to Python sample from the prevoius post.

Import TF and Keras. We'll define a network with 2 inputs and 2 outpus.

{% highlight python %}

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

num_inputs = 2
num_outputs = 2

{% endhighlight %}


Create 2 layer sequenial model

{% highlight python %}

model = tf.keras.Sequential()

# No activation
# 1.0 for weights initialization
# 2.0 for bias initialization
layer1 = Dense(units=num_inputs, use_bias=True, activation=None, weights=[np.ones([num_inputs, num_inputs]), np.full([num_inputs], 2.0)])
layer2 = Dense(units=num_outputs, use_bias=True, activation=None, weights=[np.ones([num_inputs, num_outputs]), np.full([num_outputs], 2.0)])
model.add(layer1)
model.add(layer2)

{% endhighlight %}

Use mean square error for the loss function.

{% highlight python %}
# use MSE as loss function
loss_fn = tf.keras.losses.MeanSquaredError()

{% endhighlight %}

Hardcode model input and expected model output. We'll use the same array values later in C++ implementation.

{% highlight python %}
# Arbitrary model input
x = np.array([2.0, 0.5])

# Expected output
y_true = np.array([2.0, 1.0])
{% endhighlight %}


Use Stochastic Gradient Decent (SGD) optimizer.

SGD weight update rule is
$$
W = W - LR * \nabla
$$

$$\nabla$$ is weight gradient and $$LR$$ is learning rate.

For now we'll assume learning rate equal to 1.0

{% highlight python %}
# SGD update rule for parameter w with gradient g when momentum is 0 is as follows:
#   w = w - learning_rate * g
#
#   For simplicity make learning_rate=1.0
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0)
{% endhighlight %}


In the training loop we'll compute model output for input X, compute and backpropagate the loss.

{% highlight python %}
# Get model output y for input x, compute loss, and record gradients
with tf.GradientTape(persistent=True) as tape:

    # get model output y for input x
    # add newaxis for batch size of 1
    xt = tf.convert_to_tensor(x[np.newaxis, ...])
    tape.watch(xt)
    y = model(xt)

    # loss gradient with respect to loss input y
    dy_dw = tape.gradient(y, model.trainable_variables)

    # obtain MSE loss
    loss = loss_fn(y_true, y)

    # loss gradient with respect to loss input y
    dloss_dy = tape.gradient(loss, y)

    # adjust Dense layer weights
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
{% endhighlight %}


Finally we'll print inputs, outputs, gradients, and updated Dense layer weights.

{% highlight python %}

# print model input and output excluding batch dimention
print(f"input x={x}")
print(f"output y={y[0]}")
print(f"expected output y_true={y_true}")

# print MSE loss
print(f"loss={loss}")

# print loss gradients
print("dloss_dy={}".format(*[v.numpy() for v in dloss_dy]))

# print weight gradients d_loss/d_w
print("grad=\n{}".format(*[v.numpy() for v in grad]))

# print updated dense layer weights
print("updated weights=")
print(*[v.numpy() for v in model.trainable_variables], sep="\n")

{% endhighlight %}


### After running Python example:

{% highlight bash %}
$ python3 dense3.py
input x=[2.  0.5]
output y=[11. 11.]
expected output y_true=[2. 1.]
loss=90.5
dloss_dy=[ 9. 10.]
grad=
[[38.  38. ]
 [ 9.5  9.5]]
updated weights=
0) dense/kernel:0
[[-37.  -37. ]
 [ -8.5  -8.5]]
1) dense/bias:0
[-17. -17.]
2) dense_1/kernel:0
[[-39.5 -44. ]
 [-39.5 -44. ]]
3) dense_1/bias:0
[-7. -8.]
{% endhighlight %}

Output of Python dense3.py will be used to validate the following C++ code.
Let's code the same example in C++


{% highlight c++ %}

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

{% endhighlight %}

Lambda will be used to pretty print inputs, outputs, and layer weights.

{% highlight c++ %}
/*
 * Print helper function
 */
auto print_fn = [](const float& x)  -> void {printf("%.1f ", x);};

{% endhighlight %}


Dense layer weights initializer.

{% highlight c++ %}
/*
 * Constant weight intializer
 */
const float const_one = 1.0;
const float const_two = 2.0;
template<float const& value = const_one>
constexpr auto const_initializer = []() -> float
{
  return value;
};
{% endhighlight %}


Dense layer class template includes forward() and backward() functions.
forward() was modified from the previous example to add bias in output computation.
Backward() function was modified from the previous example to update bias with input gradient.

{% highlight c++ %}
/*
 * Dense layer class template
 *
 * Parameters:
 *  num_inputs: number of inputs to Dense layer
 *  num_outputs: number of Dense layer outputs
 *  T: input, output, and weights type in the dense layer
 *  initializer: weights initializer function
 */
template<size_t num_inputs, size_t num_outputs, typename T = float,
         T (*weights_initializer)() = const_initializer<const_one>,
         T (*bias_initializer)() = const_initializer<const_two> >
struct Dense
{

  typedef array<T, num_inputs> input_vector;
  typedef array<T, num_outputs> output_vector;

  vector<input_vector> weights;
  output_vector bias;

  /*
   * Dense layer constructor
   */
  Dense()
  {
    /*
     * Create num_outputs x num_inputs weights matrix
     */
    weights.resize(num_outputs);
    for (input_vector& w: weights)
      {
        generate(w.begin(), w.end(), *weights_initializer);
      }

    /*
     * Initialize bias vector
     */
    generate(bias.begin(), bias.end(), *bias_initializer);
  }

  /*
   * Dense layer forward pass
   */
  output_vector forward(const input_vector& x)
  {
    /*
     * Check for input size mismatch
     */
    assert(x.size() == weights[0].size());

    /*
     * Layer output is dot product of input with weights
     */
    output_vector activation;
    transform(weights.begin(), weights.end(),
              bias.begin(),
              activation.begin(),
              [x](const input_vector& w, T bias)
              {
                T val = inner_product(x.begin(), x.end(), w.begin(), 0.0) + bias;
                return val;
              }
              );

    return activation;
  }

  /*
   * Dense layer backward pass
   */
  input_vector backward(input_vector& input, output_vector grad)
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
    for (auto grad_i: grad)
      {
        auto row = input;
        for_each(row.begin(), row.end(), [grad_i](T &xi){ xi *= grad_i;});
        dw.push_back(row);
      }

    /*
     * Compute backpropagated gradient
     */
    input_vector ret;
    transform(weights.begin(), weights.end(), ret.begin(),
              [grad](input_vector& w)
              {
                T val = inner_product(w.begin(), w.end(), grad.begin(), 0.0);
                return val;
              });

    /*
     * compute w = w - dw
     * assume learning rate = 1.0
     */
    transform(weights.begin(), weights.end(), dw.begin(), weights.begin(),
              [](input_vector& left, input_vector& right)
              {
                transform(left.begin(), left.end(),
                          right.begin(), left.begin(), minus<T>());
                return left;
              });

    /*
     * compute bias = bias - grad
     * assume learning rate = 1.0
     */
    transform(bias.begin(), bias.end(), grad.begin(), bias.begin(),
              [](const T& bias_i, const T& grad_i)
              {
                return bias_i - grad_i;
              });

    return ret;
  }

  /*
   * Helper function to convert Dense layer to string
   * Used for printing the layer weights an biases
   */
  operator std::string() const
  {
    std::ostringstream ret;
    ret.precision(1);

    /*
     * output weights
     */
    ret << "weights:" << std::endl;
    for (int y=0; y < weights[0].size(); y++)
      {
        for (int x=0; x < weights.size(); x++)
          {
            if (weights[x][y] >= 0)
              ret << " ";
            ret << std::fixed << weights[x][y] << " ";
          }
        ret << std::endl;
      }

    /*
     * output biases
     */
    ret << "bias:" << std::endl;
    for (auto b: bias)
      {
        if (b >= 0)
          ret << " ";
        ret << std::fixed << b << " ";
      }
    ret << std::endl;

    return ret.str();
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

{% endhighlight %}

Mean Squared Error class will need it's own forward and backward functions.


{% highlight c++ %}

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
   * Forward pass computes MSE loss for inputs y (label) and yhat (predicted)
   */
  static T forward(const array<T, num_inputs>& y, const array<T, num_inputs>& yhat)
  {
    T loss = transform_reduce(y.begin(), y.end(), yhat.begin(), 0.0, plus<T>(),
                              [](const T& left, const T& right)
                              {
                                return (left - right) * (left - right);
                              }
                              );
    return loss / num_inputs;
  }

  /*
   * Backward pass computes dloss/dy for inputs y (label) and yhat (predicted)
   *
   * loss = sum((yhat[i] - y[i])^2) / N
   *   i=0...N-1
   *   where N is number of inputs
   *
   * d_loss/dy[i] = 2 * (yhat[i] - y[i]) * (-1) / N
   * d_loss/dy[i] = 2 * (y[i] - yhat[i]) / N
   *
   */
  static array<T, num_inputs> backward(const array<T, num_inputs>& y,
                                       const array<T, num_inputs>& yhat)
  {
    array<T, num_inputs> de_dy;

    transform(y.begin(), y.end(), yhat.begin(), de_dy.begin(),
              [](const T& left, const T& right)
              {
                return 2 * (right - left) / num_inputs;
              }
              );
    return de_dy;
  }

};

{% endhighlight %}

Finally, in the main function, we'll declare input x and expecetd output y_true arrays, containing the same values as in out Python example.
Then we'll compute forward and backward passes, and print the network output and updated weights.

{% highlight c++ %}

int main(void)
{
  const int num_inputs = 2;
  const int num_outputs = 2;
  const int num_iterations = 1000;

  array<float, num_inputs> x = {2.0, 0.5};
  array<float, num_outputs> ytrue = {2.0, 1.0};

  /*
   * Create dense layer and MSE loss
   */
  Dense<num_inputs, num_outputs> dense1;
  Dense<num_inputs, num_outputs> dense2;
  MSE<num_outputs> mse_loss;

  /*
   * Compute Dense layer output y for input x
   */
  auto y1 = dense1.forward(x);
  auto y2 = dense2.forward(y1);

  /*
   * Copute MSE loss for output y and label ytrue
   */
  auto loss = mse_loss.forward(ytrue, y2);

  /*
   * Benchmark Dense layer inference latency
   */
  auto ts = high_resolution_clock::now();
  for (auto iter = 0;  iter < num_iterations; iter++)
    {
      y1 = dense1.forward(x);
      y2  = dense2.forward(y1);
    }
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
  printf("output y=");
  for_each(y2.begin(), y2.end(), print_fn);
  printf("\n");

  /*
   * Print loss for output y and label ytrue
   */
  printf("loss: %f\n", loss);

  /*
   * Compute dloss/dy gradients
   */
  auto dloss_dy = mse_loss.backward(ytrue, y2);

  /*
   * Back propagate loss
   */
  auto bw2 = dense2.backward(y1, dloss_dy);
  dense1.backward(x, bw2);

  /*
   * print dloss/dy
   */
  printf("d(loss)/dy: ");
  for_each(dloss_dy.begin(), dloss_dy.end(), print_fn);
  printf("\n");

  /*
   * Print updated Dense layer weights
   */
  printf("updated dense 1 layer weights:\n%s", ((string)dense1).c_str());
  printf("updated dense 2 layer weights:\n%s", ((string)dense2).c_str());

  /*
   * Print average latency
   */
  printf("time dt=%f usec\n", dt_us);

  return 0;
}

{% endhighlight %}


### After compiling and running C++ example:

{% highlight bash %}
 g++ -o dense3 -std=c++2a dense3.cpp && ./dense3
input x=2.0 0.5
output y=11.0 11.0
loss: 90.500000
d(loss)/dy: 9.0 10.0
updated dense 1 layer weights:
weights:
-37.0 -37.0
-8.5 -8.5
bias:
-17.0 -17.0
updated dense 2 layer weights:
weights:
-39.5 -44.0
-39.5 -44.0
bias:
-7.0 -8.0
time dt=0.199000 usec
{% endhighlight %}

As one can verify, forward path output of the C++ implementation matches the Python code.
Also, weights and biases of dense layers after backpropagation match in Python and C++ code.

Python source code for this example is at [dense3.py] [python_source_code]

C++ implementation is at [dense3.cpp] [cpp_source_code]

[previous_post]:  https://alexgl-github.github.io/github/jekyll/2021/04/28/Multiple_Dense_layers.html
[python_source_code]:  https://github.com/alexgl-github/alexgl-github.github.io/tree/main/src/dense3.py
[cpp_source_code]:  https://github.com/alexgl-github/alexgl-github.github.io/tree/main/src/dense3.cpp
