#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <array>
#include <iterator>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

/*
 * Print helper function
 */
auto print_fn = [](const float& x)  -> void {printf("%.1f ", x);};

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
    transform(weights.begin(), weights.end(), activation.begin(), [x](const input_vector& w)
              {
                T val = inner_product(x.begin(), x.end(), w.begin(), 0.0);
                return val;
              }
              );

    return activation;
  }

  /*
   * Dense layer backward pass
   */
  input_vector backward(input_vector& input, output_vector dloss_dy)
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
     * Compute backpropagated gradient
     */
    vector<output_vector> weights_transposed;
    weights_transposed.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++)
      {
        for (size_t j = 0; j < num_outputs; j++)
          {
            weights_transposed[i][j] = weights[j][i];
          }
      }

    input_vector ret;
    transform(weights_transposed.begin(), weights_transposed.end(), ret.begin(),
              [dloss_dy](input_vector& w)
              {
                T val = inner_product(w.begin(), w.end(), dloss_dy.begin(), 0.0);
                return val;
              });

    /*
     * compute dw
     * dw = outer(x, de_dy)
     */
    vector<input_vector> dw;
    for (auto dloss_dy_i: dloss_dy)
      {
        auto row = input;
        for_each(row.begin(), row.end(), [dloss_dy_i](T &xi){ xi *= dloss_dy_i;});
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
              });

    return ret;
  }

  /*
   * Helper function to convert Dense layer to string
   * Used for printing the layer
   */
  operator std::string() const
  {
    std::ostringstream ret;
    ret.precision(1);

    for (size_t y=0; y < weights[0].size(); y++)
      {
        for (size_t x=0; x < weights.size(); x++)
          {
            if (weights[x][y] >= 0)
              ret << " ";
            ret << std::fixed << weights[x][y] << " ";
          }
        ret << std::endl;
      }
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
  static array<T, num_inputs> backward(const array<T, num_inputs>& y, const array<T, num_inputs>& yhat)
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
   * Copute MSE loss for output y and labe yhat
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
  printf("output y1=");
  for_each(y1.begin(), y1.end(), print_fn);
  printf("\n");

  printf("output y2=");
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
