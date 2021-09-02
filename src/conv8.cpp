#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <array>
#include <iterator>
#include <variant>
#include <random>

using namespace std;

/*
 * Print helper function
 */
auto print_n(const float& x, const char* fmt)
{
  printf(fmt, x);
}

template<typename T>
auto print_n(const T& x, const char* fmt="%7.2f ")
{
  std::for_each(x.begin(), x.end(),
                [fmt](const auto& xi)
                {
                  print_n(xi, fmt);
                });
  printf("\n");
}


/*
 *
 */
template<typename G>
auto initialize(float& x, G& generator)
{
  x = generator();
}

template<typename T, typename G>
auto initialize(T& x, G& generator)
{
  std::for_each(x.begin(), x.end(), [generator](auto& x_i)
                {
                  initialize(x_i, generator);
                });
}


/*
 * Incremental and decremental initializers
 */
template<int initial_value = 0, typename T=float>
auto gen_inc = []() { static int i = initial_value; return static_cast<T>(i++);};

template<int initial_value = 0, typename T=float>
auto gen_dec = []() { static int i = initial_value; return static_cast<T>(i--);};

/*
 * Constant weight intializer
 */

template<typename T = float, const int num = 0, const int det = 1>
constexpr auto const_initializer = []() -> float
{
  return static_cast<T>(num) / static_cast<T>(det);
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
    return loss / static_cast<T>(num_inputs);
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

  input_type backward(const output_type& grad)
  {
    input_type grad_out;
    size_t idx = 0;

    for (size_t i = 0; i < input_height; i++)
      {
        for (size_t j = 0; j < input_width; j++)
          {
            for (size_t channel = 0; channel < channels; channel++)
              {
                grad_out[channel][i][j] = grad[idx++];
              }
          }
      }
    return grad_out;
  }
};


/*
 * 2D convolution class template
 */
template<int input_height,      /* input height */
         int input_width,       /* input width */
         int channels_inp = 1,  /* input channels */
         int channels_out =1,   /* output channels */
         int kernel_size = 3,   /* kernel size */
         int stride = 1,        /* stride (currently unused) */
         bool use_bias = false, /* enable bias flag */
         typename T = float,    /* convolution data type */
         T (*weights_initializer)() = const_initializer<>,  /* initializer function for weights */
         T (*bias_initializer)() = const_initializer<>>     /* initializer function for biases */
struct Conv2D
{
  typedef array<T, input_width> input_row;
  typedef array<input_row, input_height> input_plane;
  typedef array<input_plane, channels_inp> conv_input;

  static const int output_width = input_width;
  static const int output_height = input_height;

  typedef array<T, output_width> output_row;
  typedef array<output_row, output_height> output_plane;
  typedef array<output_plane, channels_out> conv_output;

  typedef array<array<T, kernel_size>, kernel_size> conv_kernel;
  /*
   * OIHW
   */
  typedef array<array<conv_kernel, channels_inp>, channels_out> conv_weights;

  typedef array<T, channels_out> conv_bias;

  conv_weights weights;
  conv_bias bias;
  conv_weights dw;
  conv_bias db;

  static const int pad_size = kernel_size / 2;

  Conv2D()
  {
    initialize(weights, *weights_initializer);
    initialize(bias, *bias_initializer);
    initialize(dw, const_initializer<>);
    initialize(db, const_initializer<>);
  }

  /*
   * Compute convolution of 2D inputs x and w
   */
  template<int height_x, int width_x, int height_w, int width_w>
  static T conv (const std::array<std::array<T, width_x>, height_x>& x,
          const std::array<std::array<T, width_w>, height_w>& w,
          int i, int j)
  {
    const int pad_top = (i < 0) ? (-i) : 0;
    const int pad_bot = (i > height_x - height_w) ? (i - height_x + height_w) : 0;
    const int pad_left = (j < 0) ? (- j) : 0;
    const int pad_right = (j > width_x - width_w) ? (j - width_x + width_w) : 0;

    T sum =
      std::transform_reduce(w.begin() + pad_top,
                            w.end()   - pad_bot,
                            x.begin() + pad_top + i,
                            static_cast<T>(0),
                            std::plus<T>(),
                            [j, pad_left, pad_right](auto& w_i, auto& x_i) -> T
                            {
                              return std::inner_product(w_i.begin() + pad_left,
                                                        w_i.end()   - pad_right,
                                                        x_i.begin() + pad_left + j,
                                                        static_cast<T>(0));
                            }
                            );

    return sum;
  };

  /*
   * Forward path computes convolution of input x and kernel weights w
   */
  conv_output forward(const conv_input& x)
  {
    conv_output y;

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int i = 0; i < output_height; i++)
          {
            for (int j = 0; j < output_width; j++)
              {
                y[output_channel][i][j] = use_bias * bias[output_channel];
                for (int input_channel = 0; input_channel < channels_inp; input_channel++)
                  {
                    y[output_channel][i][j] +=
                      conv<input_height,
                           input_width,
                           kernel_size,
                           kernel_size>(x[input_channel],
                                        weights[output_channel][input_channel],
                                        i - pad_size,
                                        j - pad_size);
                  }
              }
          }
      }
    return y;
  }

  /*
   * Backward path computes:
   *  - weight gradient dW
   *  - bias gradient dB
   *  - output gradient dX
   */
  conv_input backward(const conv_input& x,  const conv_output& grad)
  {
    conv_input grad_out = {};

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int i = 0; i < kernel_size; i++)
          {
            for (int j = 0; j < kernel_size; j++)
              {
                for (int input_channel = 0; input_channel < channels_inp; input_channel++)
                  {
                    dw[output_channel][input_channel][i][j] +=
                      conv<input_height,
                           input_width,
                           output_height,
                           output_width>(x[input_channel],
                                         grad[output_channel],
                                         i - pad_size,
                                         j - pad_size);
                  }
              }
          }
      }

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        db[output_channel] += std::accumulate(grad[output_channel].cbegin(),
                                              grad[output_channel].cend(),
                                              static_cast<T>(0),
                                              [](auto total, const auto& grad_i)
                                              {
                                                return std::accumulate(grad_i.cbegin(),
                                                                       grad_i.cend(),
                                                                       total);
                                              });
      }

    conv_weights weights_rot180;
    for (int input_channel = 0; input_channel < channels_inp; input_channel++)
      {
        for (int output_channel = 0; output_channel < channels_out; output_channel++)
          {

            for (size_t i = 0; i < kernel_size; i++)
              {
                for (size_t j = 0; j < kernel_size; j++)
                  {
                    weights_rot180[output_channel][input_channel][i][j] =
                      weights[output_channel][input_channel][i][kernel_size - j - 1];
                  }
              }

            for (size_t i = 0; i < kernel_size/2; i++)
              {
                for (size_t j = 0; j < kernel_size; j++)
                  {
                    auto t = weights_rot180[output_channel][input_channel][i][j];
                    weights_rot180[output_channel][input_channel][i][j] =
                      weights_rot180[output_channel][input_channel][kernel_size - i - 1][j];
                    weights_rot180[output_channel][input_channel][kernel_size - i - 1][j] = t;
                  }
              }

          }
      }

    for (int input_channel = 0; input_channel < channels_inp; input_channel++)
      {
        for (int i = 0; i < input_height; i++)
          {
            for (int j = 0; j < input_width; j++)
              {
                grad_out[input_channel][i][j] = 0.0;
                for (int output_channel = 0;
                     output_channel < channels_out;
                     output_channel++)
                  {
                    grad_out[input_channel][i][j] +=
                      conv<output_height,
                           output_width,
                           kernel_size,
                           kernel_size>(grad[output_channel],
                                        weights_rot180[output_channel][input_channel],
                                        i - pad_size,
                                        j - pad_size);
                  }
              }
          }
      }

    return grad_out;
  }

  /*
   * Apply previously computed weight and bias gradients dW, dB, reset gradients to 0
   */
  void train(float learning_rate)
  {
    /*
     * compute w = w - learning_rate * dw
     */
    for (int input_channel = 0; input_channel < channels_inp; input_channel++)
      {
        for (int output_channel = 0; output_channel < channels_out; output_channel++)
          {
            for (size_t i = 0; i < kernel_size; i++)
              {
                for (size_t j = 0; j < kernel_size; j++)
                  {
                    auto weight_update =
                      learning_rate * dw[output_channel][input_channel][i][j];
                    weights[output_channel][input_channel][i][j] -= weight_update ;
                  }
              }
          }
      }

    /*
     * compute bias = bias - learning_rate * db
     */
    if (use_bias)
      {
        for (int output_channel = 0; output_channel < channels_out; output_channel++)
          {
            bias[output_channel] -= learning_rate * db[output_channel];
          }
      }

    /*
     * Reset accumulated dw and db
     */
    reset_gradients();
  }

  /*
   * Reset accumulated weight and bias gradients
   */
  void reset_gradients()
  {
    initialize(dw, const_initializer<>);
    initialize(db, const_initializer<>);
  }

};

/*
 * DNN train and validation loops are implemented in the main() function.
 */
int main(void)
{
  const int input_height = 5;
  const int input_width = 5;
  const int channels_in = 1;
  const int channels_out = 2;
  const int kernel_size = 3;
  std::array<std::array<std::array<float, input_width>,
                        input_height>, channels_in> x = {};

  initialize(x, gen_dec<input_height * input_width * channels_in>);

  std::array<float, input_height * input_width * channels_out>  y_true;
  std::fill(y_true.begin(), y_true.end(), 1.0);

  /*
   * Create DNN layers and the loss
   */
  Conv2D<input_height,         /* input height */
         input_width,          /* input width */
         channels_in,          /* number of input channels */
         channels_out,         /* number of output channels */
         kernel_size,          /* convolution kernel size */
         1,                    /* stride */
         true,                 /* use_bias flag */
         float,                /* conv data type */
         gen_inc<1>,           /* initialier for kernel weights */
         gen_dec<channels_out> /* initialier for bias weights */
         > conv;
  Flatten<input_height, input_width, channels_out> flatten;
  MSE<input_height * input_width * channels_out> loss_fn;

  printf("input x:\n");
  print_n(x);

  printf("conv weights:\n");
  print_n(conv.weights);
  printf("conv bias:");
  print_n(conv.bias);

  auto y1 = conv.forward(x);
  auto y  = flatten.forward(y1);
  auto loss = loss_fn.forward(y_true, y);

  printf("output y:\n");
  print_n(y1);

  printf("loss: %.5f\n", loss);

  auto dloss_dy  = loss_fn.backward(y_true, y);
  auto dloss_dy1 = flatten.backward(dloss_dy);
  auto dloss_dx  = conv.backward(x, dloss_dy1);

  printf("dloss_dy:\n");
  print_n(dloss_dy1);

  printf("dloss_dx:\n");
  print_n(dloss_dx);

  conv.train(/*learning_rate */ 1.0);

  printf("updated conv weights:\n");
  print_n(conv.weights);

  printf("updated conv bias:\n");
  print_n(conv.bias);

  return 0;
}
