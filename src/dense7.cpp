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
#include "mnist.h"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using mnist_error = mnist<10, 28*28, float>::error;

/*
 * Print helper function
 */
constexpr auto print_fn = [](const float& x)  -> void {printf("%.7f ", x);};

/*
 * Constant weight intializer
 */
const float const_onehalf = 0.0;
const float const_one = 1.0;
const float const_zero = 0.0;
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
  vector<array<T*, num_outputs>> weights_transposed;

  /*
   * Default dense layer constructor
   */
  Dense(bool _use_bias=true)
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
     * Ctreate transposed array of weighst pointers
     */
    weights_transposed.resize(num_inputs);
    for (int i = 0; i < num_inputs; i++)
      {
        for (int j = 0; j < num_outputs; j++)
          {
            weights_transposed[i][j] = &weights[j][i];
          }
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
                T val = inner_product(w.begin(), w.end(), x.begin(), 0.0);
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
  input_vector backward(const input_vector& input, const output_vector grad, float learning_rate = 1.0)
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
    input_vector ret;
    transform(weights_transposed.begin(), weights_transposed.end(), ret.begin(),
              [grad](array<T*, num_outputs>& w)
              {
                T val = inner_product(w.begin(), w.end(), grad.begin(), 0.0,
                                      [](const T& l, const T& r)
                                      {
                                        return l + r;
                                      },
                                      [](const T* l, const T& r)
                                      {
                                        return *l * r;
                                      }
                                      );
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
     * compute w = w - learning_rate * dw
     */
    transform(weights.begin(), weights.end(), dw.begin(), weights.begin(),
              [learning_rate](input_vector& left, input_vector& right)
              {
                transform(left.begin(), left.end(), right.begin(), left.begin(),
                          [learning_rate](const T& w_i, const T& dw_i)
                          {
                            return w_i - learning_rate * dw_i;
                          });
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
                T out = grad_i * y_i * (1 - y_i);
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
  static input_vector backward(const input_vector& y, const input_vector& grad_inp)
  {
    input_vector grad_out;
    vector<input_vector> J;

    /*
     * Compute Jacobian of Softmax
     */
    int s_i_j = 0;
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
 *  E = - sum(yhat * log(y))
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
   * Forward pass computes CCE loss for inputs yhat (label) and y (predicted)
   */
  static T forward(const input_vector& yhat, const input_vector& y)
  {
#if 1
    T loss = transform_reduce(yhat.begin(), yhat.end(), y.begin(), 0.0, plus<T>(),
                              [](const T& yhat_i, const T& y_i)
                              {
                                return yhat_i * logf(y_i);
                              }
                              );
#else
    input_vector cce;
    transform(yhat.begin(), yhat.end(), y.begin(), cce.begin(),
              [](const T& yhat_i, const T& y_i)
              {
                return yhat_i * logf(y_i);
              }
              );
    T loss = accumulate(cce.begin(), cce.end(), 0.0);
#endif
    return -1 * loss;
  }

  /*
   * Backward pass computes dloss/dy for inputs yhat (label) and y (predicted):
   *
   * dloss/dy = - yhat/y
   *
   */
  static input_vector backward(input_vector yhat, input_vector y)
  {
    array<T, num_inputs> de_dy;

    transform(yhat.begin(), yhat.end(), y.begin(), de_dy.begin(),
              [](const T& yhat_i, const T& y_i)
              {
                return -1 * yhat_i / y_i;
              }
              );
    return de_dy;
  }

};


int main(void)
{
  const char* images_path = "./data/train-images-idx3-ubyte";
  const char* labels_path = "./data/train-labels-idx1-ubyte";
  const int num_classes = 10;
  const int num_rows = 28;
  const int num_cols = 28;
  const int image_size = num_rows * num_cols;
  std::array<float, image_size> image;
  std::array<float, num_classes> label;
  const int num_epochs = 3000;
  float learning_rate = 0.01;

  mnist dataset(images_path, labels_path);
  printf("found %d images and %d labels\n", dataset.number_of_images, dataset.number_of_labels);

  /*
   * Create DNN layers and the loss
   */
  Dense<image_size, 256, float, const_initializer<const_onehalf>, const_initializer<const_onehalf>> dense1;
  Sigmoid<256> sigmoid1;
  Dense<256, num_classes, float, const_initializer<const_onehalf>, const_initializer<const_onehalf>> dense2;
  Softmax<num_classes> softmax;
  CCE<num_classes> loss_fn;
  auto iterations = dataset.number_of_images;

  for (auto epoch=0; epoch < num_epochs; epoch++)
    {
      dataset.rewind();
      float loss_epoch = 0;
      float loss;

      auto ts = high_resolution_clock::now();

      for (auto iter = 0; iter < iterations; iter++)
        {
          auto ret1 = dataset.read_next(image, label);
          if (ret1 == mnist_error::MNIST_EOF)
            {
              break;
            }

          /*
           * Compute Dense layer output y for input x
           */
          auto y1 = dense1.forward(image);
          auto y2 = sigmoid1.forward(y1);
          auto y3 = dense2.forward(y2);
          auto y4 = softmax.forward(y3);
          auto loss = loss_fn.forward(label, y4);

          /*
           * Back propagate loss
           */
          auto dloss_dy4 = loss_fn.backward(label, y4);
          auto dy4_dy3 = softmax.backward(y4, dloss_dy4);
          auto dy3_dy2 = dense2.backward(y2, dy4_dy3, learning_rate);
          auto dy2_dy1 = sigmoid1.backward(y2, dy3_dy2);
          auto dy1_dx = dense1.backward(image, dy2_dy1, learning_rate);
          loss_epoch += loss;

          auto te = high_resolution_clock::now();
          auto dt_s = (float)duration_cast<seconds>(te - ts).count() / (iter + 1);

          if ((iter % 5000) == 0)
            {
              printf("epoch=%d/%d iter=%d/%d time/iter=%.3f sec loss: %f\n", epoch, num_epochs, iter, iterations, dt_s, loss_epoch / (iter + 1));
            }
        }

      auto te = high_resolution_clock::now();
      auto dt_s = (float)duration_cast<seconds>(te - ts).count();

      loss_epoch = loss_epoch / iterations;
      printf("epoch %d/%d time/epoch=%.5f sec time left=%.4f hr avg loss: %f\n", epoch, num_epochs, dt_s / iterations / (60 * 60), dt_s * (num_epochs - epoch), loss_epoch);

    }

  return 0;
}
