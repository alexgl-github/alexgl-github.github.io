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
   * Dense layer backward pass
   */
  void backward(const input_vector& input, const output_vector& dloss_dy)
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

    for (auto const& dloss_dyi: dloss_dy)
      {
        auto row = input;
        for_each(row.begin(), row.end(), [dloss_dyi](T &xi){ xi *= dloss_dyi;});
        dw.emplace_back(row);
      }

    /*
     * compute w = w - dw
     * assume learning rate = 1.0
     */
    transform(weights.begin(), weights.end(), dw.begin(), weights.begin(),
              [](input_vector& left, const input_vector& right)
              {
                transform(left.begin(), left.end(), right.begin(), left.begin(), minus<T>());
                return left;
              }
              );
  }

  /*
   * Helper function to convert Dense layer to string.
   * It is used for printing the layer weights.
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
  static T forward(const array<T, num_inputs>& yhat, const array<T, num_inputs>& y)
  {
    T loss = transform_reduce(yhat.begin(), yhat.end(), y.begin(), 0.0, plus<T>(),
                              [](const T& left, const T& right)
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
  static array<T, num_inputs> backward(const array<T, num_inputs>& yhat, const array<T, num_inputs>& y)
  {
    array<T, num_inputs> de_dy;

    transform(yhat.begin(), yhat.end(), y.begin(), de_dy.begin(),
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
