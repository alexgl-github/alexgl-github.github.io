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
#include <cmath>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

/*
 * Print helper function
 */
constexpr auto print_fn = [](const float& x)  -> void {printf("%.7f ", x);};

/*
 * Constant weight intializer
 */
const float const_zero = 0.0;
const float const_one = 1.0;
const float const_two = 2.0;
template<float const& value = const_one>
constexpr auto const_initializer = []() -> float
{
  return value;
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
template<size_t num_inputs, size_t num_outputs, typename T = float, T (*weights_initializer)() = const_initializer<const_zero>, T (*bias_initializer)() = const_initializer<const_zero> >
struct Dense
{

  typedef array<T, num_inputs> input_vector;
  typedef array<T, num_outputs> output_vector;
  typedef T (*initializer)();
  vector<input_vector> weights;
  output_vector bias;
  bool use_bias = true;

  /*
   * Default dense layer constructor
   */
  Dense(bool _use_bias)
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
    use_bias = _use_bias;
    generate(bias.begin(), bias.end(), *bias_initializer);
  }

  /*
   * Dense layer constructor from provided weigths and biases
   * Note: weights are stored transposed
   */
  Dense(const array<array<T, num_inputs>, num_outputs>& weights_init, const array<T, num_outputs>& biases_init)
  {
    /*
     * Create num_outputs x num_inputs weights matrix
     */
    for (auto weights_row: weights_init)
      {
        weights.push_back(weights_row);
      }

    /*
     * Initialize bias vector
     */
    bias = biases_init;
  }

  /*
   * Dense layer constructor from provided weigths. Bias is not used
   * Note: weights are stored transposed
   */
  Dense(const array<array<T, num_inputs>, num_outputs>& weights_init)
  {
    /*
     * Create num_outputs x num_inputs weights matrix
     */
    for (auto weights_row: weights_init)
      {
        weights.push_back(weights_row);
      }

    use_bias = false;
  }

  /*
   * Dense layer forward pass
   * Computes X * W + B
   *  X - input row vector
   *  W - weights matrix
   *  B - bias row wector
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
    transform(weights.begin(), weights.end(), bias.begin(), activation.begin(), [x, this](const input_vector& w, T bias)
              {
                T val = inner_product(x.begin(), x.end(), w.begin(), 0.0);
                if (use_bias)
                  {
                    val += bias;
                  }
                return val;
              }
              );

    return activation;
  }

  /*
   * Dense layer backward pass
   */
  input_vector backward(const input_vector& input, const output_vector grad)
  {
    /*
     * Weight update according to SGD algorithm with momentum = 0.0 is:
     *  w = w - learning_rate * d_loss/dw
     *
     * For simplicity assume learning_rate = 1.0
     *
     * d_loss/dw = dloss/dy * dy/dw
     * d_loss/dbias = dloss/dy * dy/dbias
     *
     * dloss/dy is input gradient grad
     *
     * dy/dw is :
     *  y = w[0]*x[0] + w[1] * x[1] +... + w[n] * x[n] + bias
     *  dy/dw[i] = x[i]
     *
     * dy/dbias is :
     *  dy/dbias = 1
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
              [grad](output_vector& w)
              {
                T val = inner_product(w.begin(), w.end(), grad.begin(), 0.0);
                return val;
              });

    /*
     * compute dw
     * dw = outer(x, grad)
     */
    vector<input_vector> dw;
    for (auto grad_i: grad)
      {
        auto row = input;
        for_each(row.begin(), row.end(), [grad_i](T &xi){ xi *= grad_i;});
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

    if (use_bias)
      {
        /*
         * compute bias = bias - grad
         * assume learning rate = 1.0
         */
        transform(bias.begin(), bias.end(), grad.begin(), bias.begin(),
                  [](const T& bias_i, const T& grad_i)
                  {
                    return bias_i - grad_i;
                  });
      }

    return ret;
  }

  /*
   * Helper function to convert Dense layer to string
   * Used for printing the layer weights and biases
   */
  operator std::string() const
  {
    std::ostringstream ret;
    ret.precision(7);

    /*
     * output weights
     */
    ret << "weights:" << std::endl;
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

    if (use_bias)
      {
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
 * Sigmoid layer class template
 */
template<size_t num_inputs, typename T = float>
struct Sigmoid
{
  typedef array<T, num_inputs> input_vector;

  static input_vector forward(const input_vector& y)
  {
    input_vector ret;

    transform(y.begin(), y.end(), ret.begin(),
              [](const T& yi)
              {
                T out = 1.0  / (1.0 + expf(-yi));
                return out;
              });
    return ret;
  }

  static input_vector backward(const input_vector& y, const input_vector grad)
  {
    input_vector ret;

    transform(y.begin(), y.end(), grad.begin(), ret.begin(),
              [](const T& y_i, const T& grad_i)
              {
                T s = 1.0  / (1.0 + expf(-y_i));
                T out = grad_i * s * (1 - s);
                return out;
              });
    return ret;
  }

};


/*
 * Softmax layer class template
 */
template<size_t num_inputs, typename T = float>
struct Softmax
{
  typedef array<T, num_inputs> input_vector;

  /*
   * Softmax forward function
   */
  static input_vector forward(const input_vector& x)
  {
    input_vector y;

    /*
     * compute exp(x_i) / sum(exp(x_i), i=1..N)
     */
    transform(x.begin(), x.end(), y.begin(),
              [](const T& yi)
              {
                T out = expf(yi);
                return out;
              });

    T sum = accumulate(y.begin(), y.end(), static_cast<T>(0));

    for_each(y.begin(), y.end(), [sum](T &yi){ yi /= sum;});

    return y;
  }

  /*
   * Softmax backward function
   */
  static input_vector backward(const input_vector& x, const input_vector& grad_inp)
  {
    input_vector grad_out;
    input_vector y;
    vector<input_vector> J;

    y = forward(x);
    int s_i_j = 0;

    /*
     * Compute Jacobian of Softmax
     */
    for (auto y_i: y)
      {
        auto row = y;
        for_each(row.begin(), row.end(), [y_i](T& y_j){ y_j = -y_i * y_j;});
        row[s_i_j] += y_i;
        s_i_j++;
        J.push_back(row);
      }

    /*
     * Compute dot product of gradient and Softmax Jacobian
     */
    transform(J.begin(), J.end(), grad_out.begin(),
              [grad_inp](const input_vector& j)
              {
                T val = inner_product(j.begin(), j.end(), grad_inp.begin(), 0.0);
                return val;
              }
              );

    return grad_out;

  }

};


/*
 * Categorical Crossentropy loss
 *
 * Forward:
 *  E = - sum(y * log(yhat))
 *
 * Parameters:
 *  num_inputs: number of inputs to loss function.
 *  T: input type, float by defaut.
 */
template<size_t num_inputs, typename T = float>
struct CCE
{

  typedef array<T, num_inputs> input_vector;

  /*
   * Forward pass computes CCE loss for inputs y (label) and yhat (predicted)
   */
  static T forward(const input_vector& y, const input_vector& yhat)
  {
    T loss = transform_reduce(y.begin(), y.end(), yhat.begin(), 0.0, plus<T>(),
                              [](const T& y_i, const T& yhat_i)
                              {
                                return y_i * logf(yhat_i);
                              }
                              );
    return -1 * loss;
  }

  /*
   * Backward pass computes dloss/dy for inputs yhat (label) and y (predicted):
   *
   * dloss/dy = - yhat/y
   *
   */
  static input_vector backward(const input_vector& y, const input_vector& yhat)
  {
    array<T, num_inputs> de_dy;

    transform(y.begin(), y.end(), yhat.begin(), de_dy.begin(),
              [](const T& y_i, const T& yhat_i)
              {
                return -1 * y_i / yhat_i;
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

  array<float, num_inputs> x = {-1.0, 1.0, 2.0};
  array<float, num_outputs> ytrue = {1.0, 0.0};
  array<float, num_outputs> biases_init = {1.0, 2.0};
  array<array<float, num_outputs>, num_outputs> weights_init =
    {
      {
        {1.0, 3.0},
        {2.0, 2.0}
      }
    };

  /*
   * Create DNN layers and the loss
   */
  Dense<num_inputs, num_outputs, float, const_initializer<const_one>, const_initializer<const_two> > dense1(true);
  Sigmoid<num_outputs> sigmoid;
  Dense<num_outputs, num_outputs, float, const_initializer<const_one> > dense2(weights_init, biases_init);
  Softmax<num_outputs> softmax;
  CCE<num_outputs> loss_fn;

  /*
   * Compute Dense layer output y for input x
   */
  auto y1 = dense1.forward(x);
  auto y2 = sigmoid.forward(y1);
  auto y3 = dense2.forward(y2);
  auto y4 = softmax.forward(y3);

  /*
   * Copute loss for output y and label ytrue
   */
  auto loss = loss_fn.forward(ytrue, y4);

  /*
   * Benchmark Dense layer inference latency
   */
  auto ts = high_resolution_clock::now();
  for (auto iter = 0;  iter < num_iterations; iter++)
    {
      y1 = dense1.forward(x);
      y2 = sigmoid.forward(y1);
      y3 = dense2.forward(y2);
      y4 = softmax.forward(y3);
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
   * Print DNN output y4 and expected output ytrue
   */
  printf("output y=");
  for_each(y4.begin(), y4.end(), print_fn);
  printf("\n");

  printf("expected output ytrue=");
  for_each(ytrue.begin(), ytrue.end(), print_fn);
  printf("\n");

  /*
   * Print loss for output y and label ytrue
   */
  printf("loss: %f\n", loss);

  /*
   * Back propagate loss
   */
  auto dloss_dy4 = loss_fn.backward(ytrue, y4);
  auto dy4_dy3 = softmax.backward(y3, dloss_dy4);
  auto dy3_dy2 = dense2.backward(y2, dy4_dy3);
  auto dy2_dy1 = sigmoid.backward(y1, dy3_dy2);
  dense1.backward(x, dy2_dy1);

  printf("dy2=");
  for_each(dy2_dy1.begin(), dy2_dy1.end(), print_fn);
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
