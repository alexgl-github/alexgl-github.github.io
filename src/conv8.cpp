#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <array>
#include <iterator>
#include <variant>
#include <random>
#include <type_traits>


using namespace std;

/*
 * N-dimentional array
 */
template <typename T, std::size_t N, std::size_t... Dims>
struct ndarray
{

  /*
   * Array type and size
   */
  using type = std::array<typename ndarray<T, Dims...>::type, N>;
  static constexpr std::size_t size = {(Dims * ... * N)};

  /*
   * Get array value at index
   */
  static const T& ndarray_at(const typename ndarray<T, N, Dims...>::type & x, size_t index)
  {
    auto size = ndarray<T, Dims...>::size;
    size_t outer_index = index / size;
    size_t inner_index = index % size;
    return ndarray<T, Dims...>::ndarray_at(x[outer_index], inner_index);
  }

  /*
   * Reference to  array value at index
   */
  static T& ndarray_at(typename ndarray<T, N, Dims...>::type & x, size_t index)
  {
    /*
     * For explanation, see "Avoid Duplication in const and Non-const Member Function"
     * Effective C++ by Scott Meyers
     */
    return const_cast<T&>(ndarray_at(static_cast<const ndarray<T, N, Dims...>::type& >(x), index));
  }

};

/*
 * 1 dimentional array definition
 * terminating N-dimentional array recursive template
 */
template <typename T, size_t N>
struct ndarray<T, N> {

  using type = std::array<T, N>;
  static constexpr std::size_t size = {N};

  static const T& ndarray_at(const std::array<T, N>& x, size_t index)
  {
    return x[index];
  }

  static T& ndarray_at(std::array<T, N>& x, size_t index)
  {
    return const_cast<T&>(ndarray_at(x, index));
  }

};


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
 * Initializer for N-dimentional container using
 * values provided by generator G
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
 * Incremental and decremental generators
 */
template<int initial_value = 0, typename T=float>
auto gen_inc = []() { static int i = initial_value; return static_cast<T>(i++);};

template<int initial_value = 0, typename T=float>
auto gen_dec = []() { static int i = initial_value; return static_cast<T>(i--);};

/*
 * Constant weight generator
 */

template<typename T = float, const int num = 0, const int det = 1>
constexpr auto const_initializer = []() -> float
{
  return static_cast<T>(num) / static_cast<T>(det);
};

/*
 * Random uniform generator
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



template<typename T = float,
         std::size_t channels = 1,
         std::size_t... Dims>
struct Flatten
{

  typedef ndarray<float, channels, Dims...> input_array;
  typedef input_array::type input_type;
  typedef array<T, input_array::size> output_type;
  static constexpr std::size_t size_inner = {(Dims * ...)};

  output_type forward(const input_type& x)
  {
    output_type y;
    for (size_t i = 0; i < channels; i++)
      {
        for (size_t j = 0; j < size_inner; j++)
          {
            y[i+j*channels] = ndarray<float, Dims...>::ndarray_at(x[i], j);
          }
      }
    return y;
  }

  input_type backward(const output_type& y)
  {
    input_type x;

    for (size_t i = 0; i < channels; i++)
      {
        for (size_t j = 0; j < size_inner; j++)
          {
            ndarray<float, Dims...>::ndarray_at(x[i], j) = y[i+j*channels];
          }
      }

    return x;
  }
};


/*
 * 2D convolution class template
 */
template<int channels_out,  /* output channels */
         int channels_inp,   /* input channels */
         int input_height,      /* input height */
         int input_width,       /* input width */
         int kernel_size = 3,   /* kernel size */
         int stride = 1,        /* stride (currently unused) */
         bool use_bias = false, /* enable bias flag */
         typename T = float,    /* convolution data type */
         T (*weights_initializer)() = const_initializer<>,  /* initializer function for weights */
         T (*bias_initializer)() = const_initializer<>>     /* initializer function for biases */
struct Conv2D
{
  /*
   * Conv layer input
   */
  typedef ndarray<T, channels_inp, input_height, input_width> input_array;
  typedef input_array::type conv_input;

  /*
   * Conv layer output
   * Assume output dimention is the same as input dimention
   * This is equivalet to padding="same" in Keras
   */
  static const int output_width = input_width;
  static const int output_height = input_height;
  typedef ndarray<T, channels_out, output_height, output_width> output_array;
  typedef output_array::type conv_output;

  /*
   * Input zero padding required for "same" convolution padding mode.
   */
  static const int pad_size = kernel_size / 2;
  static constexpr size_t ndims = 2;
  using ssize_t = std::make_signed_t<std::size_t>;

  /*
   * Weights are in Output, Input, Height, Width (OIHW) format
   * dw is weights gradient
   */
  typedef ndarray<T, channels_out, channels_inp, kernel_size, kernel_size>::type conv_weights;
  conv_weights weights;
  conv_weights dw;

  /*
   * Bias is 1D vector
   * db is bias gradient
   */
  typedef array<T, channels_out> conv_bias;
  conv_bias bias;
  conv_bias db;

  /*
   * Default convolution constructor
   */
  Conv2D()
  {
    initialize(weights, *weights_initializer);
    initialize(bias, *bias_initializer);
    initialize(dw, const_initializer<>);
    initialize(db, const_initializer<>);
  }

  /*
   * Compute convolution of N-D inputs x and w
   */
  template<int pad_size,
           std::size_t size_x,
           std::size_t size_w>
  static T conv(const std::array<T, size_x>& x,
                 const std::array<T, size_w>& w,
                 const int idx)
  {
    const int i = idx - pad_size;
    const int pad_left  = (i < 0) ? (- i) : 0;
    const int pad_right =
      (i > static_cast<int>(size_x) - static_cast<int>(size_w)) ?
      (i - static_cast<int>(size_x) + static_cast<int>(size_w)) : 0;
    return std::inner_product(w.begin() + pad_left,
                              w.end()   - pad_right,
                              x.begin() + pad_left + i,
                              static_cast<T>(0));
  }

  template<int pad_size,
           typename type_x, std::size_t size_x,
           typename type_w, std::size_t size_w,
           typename... Idx>
  static T conv(const std::array<type_x, size_x>& x,
                const std::array<type_w, size_w>& w,
                const int idx_outer,
                Idx... idx_inner)
  {
    const int i = idx_outer - pad_size;
    const int pad_left  = (i < 0) ? (- i) : 0;
    const int pad_right =
      (i > static_cast<int>(size_x) - static_cast<int>(size_w)) ?
      (i - static_cast<int>(size_x) + static_cast<int>(size_w)) : 0;

    T sum = 0;
    for (ssize_t k = pad_left; k < (static_cast<int>(size_w) - pad_right); k ++)
      {
        sum += conv<pad_size>(x[i+k], w[k], idx_inner...);
      }

    return sum;
  };

  /*
   * Forward path computes 2D convolution of input x and kernel weights w
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
              }
          }
      }

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int input_channel = 0; input_channel < channels_inp; input_channel++)
          {
            for (int i = 0; i < output_height; i++)
              {
                for (int j = 0; j < output_width; j++)
                  {
                    y[output_channel][i][j] +=
                      conv<pad_size>(x[input_channel],
                                     weights[output_channel][input_channel],
                                     i, j);
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
                      conv<pad_size>(x[input_channel],
                                         grad[output_channel],
                                         i, j);
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
                      conv<pad_size>(grad[output_channel],
                                        weights_rot180[output_channel][input_channel],
                                        i,
                                        j);
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
  const int channels_in = 2;
  const int channels_out = 1;
  const int kernel_size = 3;

  using input_type = ndarray<float, channels_in, input_height, input_width>;
  input_type::type x = {};

  initialize(x, gen_dec<input_height * input_width * channels_in>);
  std::array<float, input_height * input_width * channels_out>  y_true;
  std::fill(y_true.begin(), y_true.end(), 1.0);

  /*
   * Create DNN layers and the loss
   */
  Conv2D<channels_out,         /* number of output channels */
         channels_in,          /* number of input channels */
         input_height,         /* input height */
         input_width,          /* input width */
         kernel_size,          /* convolution kernel size */
         1,                    /* stride */
         true,                 /* use_bias flag */
         float,                /* conv data type */
         gen_inc<1>,           /* initialier for kernel weights */
         gen_dec<channels_out> /* initialier for bias weights */
         > conv;
  Flatten<float, channels_out, input_height, input_width> flatten;
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
  print_n(flatten.backward(y));

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
