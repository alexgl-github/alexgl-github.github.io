#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <iterator>
#include <variant>
#include <random>
#include "mnist.h"
#include "f1.h"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;

/*
 * Print helper function
 */
constexpr auto print_fn = [](const float& x)  -> void {printf("%7.4f ", x);};

/*
 * Constant weight intializer
 */
const float const_one = 1.0;
const float const_zero = 0.0;
const float const_one_half = 0.0;

template<float const& value = const_zero>
constexpr auto const_initializer = []() -> float
{
  return value;
};

/*
 * Random uniform weights initializer
 */
constexpr auto random_uniform_initializer = []() -> float
{
  /*
   * Return random values in the range [-1.0, 1.0]
   */
  return 2.0 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0;
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
template<size_t num_inputs, size_t num_outputs, typename T = float, T (*weights_initializer)() = random_uniform_initializer, T (*bias_initializer)() = const_initializer<const_zero> >
struct Dense
{

  /*
   * input outut vector type definitions
   */
  typedef array<T, num_inputs> input_vector;
  typedef array<T, num_outputs> output_vector;

  /*
   * dense layer weights matrix W, used in y = X * W
   * weights are transposed to speed up forward() computation
   */
  std::array<input_vector, num_outputs> weights;

  /*
   * bias vector
   */
  output_vector bias;

  /*
   * Flag to enable/disable bias
   */
  bool use_bias = true;

  /*
   * Matrix of transposed weights W pointers, used to speed up backward() path
   */
  std::array<array<T*, num_outputs>, num_inputs> weights_transposed;

  /*
   * dw is accumulating weight updates in backward() pass.
   */
  std::array<input_vector, num_outputs> dw;

  /*
   * db is accumulating bias updates in backward() pass.
   */
  output_vector db;

  /*
   * x is for saving input to forward() call, used later in the backward() pass.
   */
  input_vector x;

  /*
   * Default dense layer constructor
   */
  Dense(bool _use_bias=true)
  {
    /*
     * Create num_outputs x num_inputs weights matrix
     */
    for (input_vector& w: weights)
      {
        generate(w.begin(), w.end(), *weights_initializer);
      }

    /*
     * Ctreate transposed array of weighst pointers
     */
    //weights_transposed.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++)
      {
        for (size_t j = 0; j < num_outputs; j++)
          {
            weights_transposed[i][j] = &weights[j][i];
          }
      }

    /*
     * Initialize bias vector
     */
    use_bias = _use_bias;
    generate(bias.begin(), bias.end(), *bias_initializer);

    /*
     * Initialize dw, db
     */
    reset_gradients();
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
  output_vector forward(const input_vector& input_x)
  {
    /*
     * Save the last input X
     */
    x = input_x;

    /*
     * Check for input size mismatch
     */
    assert(x.size() == weights[0].size());

    /*
     * Layer output is dot product of input with weights
     */
    output_vector y;
    transform(weights.begin(), weights.end(), bias.begin(), y.begin(), [this](const input_vector& w, T bias)
              {
                T y_i = inner_product(w.begin(), w.end(), x.begin(), 0.0);
                if (use_bias)
                  {
                    y_i += bias;
                  }
                return y_i;
              }
              );

    return y;
  }


  /*
   * Dense layer backward pass
   */
  input_vector backward(const output_vector& grad)
  {
    /*
     * Weight update according to SGD algorithm with momentum = 0.0 is:
     *  w = w - learning_rate * d_loss/dw
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
     *  first compute dw
     *  second update weights by subtracting dw
     */

    /*
     * Compute backpropagated gradient
     */
    input_vector grad_out;
    transform(weights_transposed.begin(), weights_transposed.end(), grad_out.begin(),
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
     * accumulate weight updates
     * compute dw = dw + outer(x, grad)
     */
    transform(dw.begin(), dw.end(), grad.begin(), dw.begin(),
              [this](input_vector& left, const T& grad_i)
              {
                /* compute outer product for each row */
                auto row = x;
                for_each(row.begin(), row.end(), [grad_i](T &xi){ xi *= grad_i;});

                /* accumulate into dw */
                std::transform (left.begin(), left.end(), row.begin(), left.begin(), std::plus<T>());
                return left;
              });
    /*
     * accumulate bias updates
     */
    std::transform(db.begin(), db.end(), grad.begin(), db.begin(), std::plus<T>());

    return grad_out;
  }

  /*
   * Update Dense layer weigts/biases using accumulated dw/db
   */
  void train(float learning_rate)
  {
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

    /*
     * compute bias = bias - learning_rate * db
     */
    if (use_bias)
      {
        /*
         * compute bias = bias - grad
         */
        transform(bias.begin(), bias.end(), db.begin(), bias.begin(),
                  [learning_rate](const T& bias_i, const T& db_i)
                  {
                    return bias_i - learning_rate * db_i;
                  });
      }

    /*
     * Reset accumulated dw and db
     */
    reset_gradients();
  }

  /*
   * Reset weigth and bias gradient accumulators
   */
  void reset_gradients()
  {
    for (input_vector& dw_i: dw)
      {
        std::fill(std::begin(dw_i), std::end(dw_i), 0.0);
      }
    std::fill(std::begin(db), std::end(db), 0.0);
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
  typedef array<T, num_inputs> output_vector;

  /*
   * x is for saving input to forward() call, used later in the backward() pass.
   */
  input_vector x;

  /*
   * Sigmoid forward pass
   */
  output_vector forward(const input_vector& input_x)
  {
    output_vector y;
    x = input_x;
    transform(x.begin(), x.end(), y.begin(),
              [](const T& yi)
              {
                T out = 1.0  / (1.0 + expf(-yi));
                return out;
              });
    return y;
  }

  /*
   * Sigmoid backward pass
   */
  input_vector backward(const output_vector grad)
  {
    input_vector grad_output;

    const output_vector y = forward(x);
    transform(y.begin(), y.end(), grad.begin(), grad_output.begin(),
              [](const T& y_i, const T& grad_i)
              {
                T out = grad_i * y_i * (1 - y_i);
                return out;
              });
    return grad_output;
  }

  /*
   * No trainabele weights in Sigmoid
   */
  void train(float lr)
  {
  }

};


/*
 * Softmax layer class template
 */
template<size_t num_inputs, typename T = float>
struct Softmax
{
  typedef array<T, num_inputs> input_vector;
  typedef array<T, num_inputs> output_vector;

  /*
   * x is for saving input to forward() call, and is used in the backward() pass.
   */
  input_vector x;

  /*
   * Softmax forward function
   */
  output_vector forward(const input_vector& input_x)
  {
    output_vector y;

    /*
     * Save input to forward()
     */
    x = input_x;

    /*
     * compute exp(x_i) / sum(exp(x_i), i=1..N)
     */
    transform(x.begin(), x.end(), y.begin(),
              [](const T& yi)
              {
                T out = expf(yi);
                return out;
              });

    T sum = accumulate(y.begin(), y.end(), static_cast<T>(0.0));
    for_each(y.begin(), y.end(), [sum](T &yi){ yi /= sum;});

    return y;
  }

  /*
   * Softmax backward function
   */
  input_vector backward(const output_vector& grad_inp)
  {
    input_vector grad_out;
    vector<input_vector> J;

    /*
     * Compute Jacobian of Softmax
     */
    const input_vector y = forward(x);
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

  /*
   * No trainable weights in softmax
   */
  void train(float lr)
  {
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
    input_vector cce;
    transform(yhat.begin(), yhat.end(), y.begin(), cce.begin(),
              [](const T& yhat_i, const T& y_i)
              {
                return yhat_i * logf(y_i);
              }
              );
    T loss = accumulate(cce.begin(), cce.end(), 0.0);
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
                return -1 * yhat_i / (y_i);
              }
              );
    return de_dy;
  }

};


/*
 * Sequential class template
 *
 *  Creates DNN layer from template list
 *  Implements higher level forward(), backward() and train() functions
 */
template<typename... T>
struct Sequential
{
  /*
   * layers array hosld DN layer objects
   */
  std::array<std::variant<T...>, sizeof...(T)> layers;

  /*
   * Sequential constructor
   */
  Sequential()
  {
    /*
     * Create DNN layers from the template list
     */
    auto create_layers = [this]<std::size_t... I>(std::index_sequence<I...>)
      {
        (void(layers[I].template emplace<I>(T())),...);
      };
    create_layers(std::make_index_sequence <sizeof...(T)>());
  }

  /*
   * Sequential forward pass will call each layer forward() function
   */
  template<size_t index=sizeof...(T)-1, typename Tn>
  auto forward(Tn& x)
  {
    if constexpr(index == 0)
       {
         return std::get<index>(layers[index]).forward(x);
       }
    else
      {
	auto y_prev = forward<index-1>(x);
	return std::get<index>(layers[index]).forward(y_prev);
      }
  }


  /*
   * Sequential backward pass will call each layer backward() function
   */
  template<size_t index=0, typename Tn>
  auto backward(Tn& dy)
  {
    if constexpr(index == sizeof...(T)-1)
       {
         return std::get<index>(layers[index]).backward(dy);
       }
    else
      {
	auto dy_prev = backward<index+1>(dy);
	return std::get<index>(layers[index]).backward(dy_prev);
      }
  }

  /*
   * Sequential class train() invokes each layer train() function
   */
  void train(float learning_rate)
  {
    [this, learning_rate]<std::size_t... I> (std::index_sequence<I...>)
      {
	(void(std::get<I>(layers[I]).train(learning_rate)), ...);
      }(std::make_index_sequence <sizeof...(T)>());
  }
};

/*
 *
 */
template<int input_height,
         int input_width,
         int channels_in = 1,
         int channels_out =1,
         int kernel_size = 3,
         int stride = 1,
         int use_bias = 1,
         typename T = float,
         T (*weights_initializer)() = const_initializer<const_one>,
         T (*bias_initializer)() = const_initializer<const_one>>
struct Conv2D
{
  typedef array<T, input_width> input_row;
  typedef array<input_row, input_height> input_plane;
  typedef array<input_plane, channels_in> conv_input;

  static const int output_width = input_width;
  static const int output_height = input_height;

  typedef array<T, output_width> output_row;
  typedef array<output_row, output_height> output_plane;
  typedef array<output_plane, channels_out> conv_output;

  typedef array<array<T, kernel_size>, kernel_size> conv_kernel;
  typedef array<array<conv_kernel, channels_in>, channels_out> conv_weights;

  typedef array<T, channels_out> conv_bias;

  conv_weights weights;
  conv_bias bias;

  static const size_t pad_size = kernel_size / 2;

  Conv2D()
  {
    for (int channel_out = 0; channel_out < channels_out; channel_out++)
      {
        for (int channel_in = 0; channel_in < channels_in; channel_in++)
          {
            for (auto& weights_row: weights[channel_out][channel_in])
              {
                std::generate(weights_row.begin(), weights_row.end(),
                              [] {
                                static int i = 1;
                                return i++;
                              });
              }
          }
      }

    if (use_bias)
      {
        generate(bias.begin(), bias.end(), *bias_initializer);
      }
  }

  conv_output forward(const conv_input& x)
  {
    conv_output y;

    auto conv = [](const input_plane& inp, const conv_kernel& w, size_t y, size_t x) -> T
      {
        const size_t pad_top = (y < pad_size) ? (pad_size - y) : 0;
        const size_t pad_bot = (y > (output_height - pad_size - 1)) ? (y - (output_height - pad_size - 1)) : 0;
        const size_t pad_left = (x < pad_size) ? (pad_size - x) : 0;
        const size_t pad_right = (x > (output_width - pad_size - 1)) ? (x - (output_width - pad_size - 1)) : 0;

        T sum = static_cast<T>(0);
        for (size_t i = pad_top; i < kernel_size-pad_bot; i++)
          {
            sum += transform_reduce(w[i].begin() + pad_left,
                                    w[i].end() - pad_right,
                                    inp[y + i - kernel_size/2].begin() + x + pad_left - kernel_size/2,
                                    0.0,
                                    std::plus<T>(),
                                    std::multiplies<T>());
          }
        return sum;
      };

    for (size_t output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (size_t i = 0; i < output_height; i++)
          {
            for (size_t j = 0; j < output_width; j++)
              {
                y[output_channel][i][j] = use_bias * bias[output_channel];
                for (size_t input_channel = 0; input_channel < channels_in; input_channel++)
                  {
                    y[output_channel][i][j] += conv(x[input_channel], weights[output_channel][input_channel], i, j);
                  }
              }
          }
      }
    return y;
  }

  conv_input backward(const conv_input& x,  const conv_output& grad)
  {
    conv_input grad_out;
    return grad_out;
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

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int input_channel = 0; input_channel < channels_in; input_channel++)
          {
            for (int y=0; y < kernel_size; y++)
              {
                for (int x=0; x < kernel_size; x++)
                  {
                    if (weights[output_channel][input_channel][y][x] >= 0)
                      ret << " ";
                    ret << std::fixed << weights[output_channel][input_channel][y][x] << " ";
                  }
                ret << std::endl;
              }
          }
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
  friend ostream& operator<<(ostream& os, const Conv2D& conv)
  {
    os << (string)conv;
    return os;
  }


};


template<std::size_t input_height,
         std::size_t input_width,
         std::size_t channels,
         typename T = float>
struct Flatten
{
  typedef array<T, input_width> input_row;
  typedef array<input_row, input_height> input_plane;
  typedef array<input_plane, channels> input_type;

  typedef array<T, input_width * input_height * channels> output_type;

  output_type forward(const input_type& x)
  {
    output_type y;
    size_t idx = 0;

    for (size_t i = 0; i < input_height; i++)
      {
        for (size_t j = 0; j < input_width; j++)
          {
            for (size_t channel = 0; channel < channels; channel++)
              {
                y[idx++] = x[channel][i][j];
              }
          }
      }
    return y;
  }
};


/*
 * DNN train and validation loops are implemented in the main() function.
 */
int main(void)
{
  const int input_height = 10;
  const int input_width = 10;
  const int channels_in = 1;
  const int channels_out = 1;
  const int kernel_size = 3;
  std::array<std::array<std::array<float, input_width>, input_height>, channels_in> x = {};

  static int i = 1;
  for (auto channel_in = 0; channel_in < channels_in; channel_in++)
    {
      for (auto& row: x[channel_in])
        {
          std::generate(row.begin(), row.end(),
                        [] {
                          return i++;
                        });
        }
    }

  std::array<float, input_height * input_width * channels_out>  y_true;
  std::fill(y_true.begin(), y_true.end(), 1.0);

  /*
   * Create DNN layers and the loss
   */
  Conv2D<input_height, input_width, channels_in, channels_out, kernel_size> conv;
  Flatten<input_height, input_width, channels_out> flatten;
  MSE<input_height * input_width * channels_out> loss_fn;
  auto y1 = conv.forward(x);
  auto y2 = flatten.forward(y1);
  auto loss = loss_fn.forward(y_true, y2);

  printf("input x=\n");
  for_each(x.begin(), x.end(),
           [](auto& x_channel)
           {
             for_each(x_channel.begin(), x_channel.end(),
                      [](auto x_row)
                      {
                        for_each(x_row.begin(), x_row.end(), print_fn);
                        printf("\n");
                      });
             printf("\n");
           });
  printf("\n");

  printf("dense layer weights:\n%s", ((std::string)conv).c_str());

  printf("output y=\n");
  for_each(y1.begin(), y1.end(),
           [](auto& y_channel)
           {
             for_each(y_channel.begin(), y_channel.end(),
                      [](auto y_row)
                      {
                        for_each(y_row.begin(), y_row.end(), print_fn);
                        printf("\n");
                      });
             printf("\n");
           });
  printf("\n");

  printf("output y flat=\n");
  for_each(y2.begin(), y2.end(), print_fn);
  printf("\n");

  printf("expected output y=\n");
  for_each(y_true.begin(), y_true.end(), print_fn);
  printf("\n");

  printf("loss=%.5f\n", loss);

  return 0;
}
