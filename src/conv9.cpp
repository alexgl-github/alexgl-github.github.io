#include <cstdio>
#include <array>
#include <variant>
#include <cmath>
#include "cifar.h"
#include "ndarray.h"
#include "f1.h"

using namespace std;

/*
 * Random uniform weights initializer
 */
template <typename T=float>
constexpr auto random_uniform_initializer = [](T& x) -> void
{
  /*
   * Return random values in the range [-0.1, 0.1]
   */
  x = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5) / 5.0;
};


/*
 * Constant value initializer
 */
template <typename T=float>
struct generate_const
{
  T value = 0;
  generate_const(T init_value=0)
  {
    value = init_value;
  }
  void operator()(T& x)
  {
    x = value;
  }
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
template<size_t num_inputs,
         size_t num_outputs,
         bool use_bias = true,
         typename T=float,
         void (*weights_initializer)(T&) = random_uniform_initializer<>,
         void (*bias_initializer)(T&) = random_uniform_initializer<> >
struct Dense
{

  /*
   * input outut vector type definitions
   */
  typedef std::array<T, num_inputs> input_vector;
  typedef std::array<T, num_outputs> output_vector;

  /*
   * dense layer weights matrix W, used in y = X * W
   */
  std::array<input_vector, num_outputs> weights;

  /*
   * bias vector
   */
  output_vector bias;

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
  Dense()
  {
    /*
     * Initialize weights and biases
     */
    for_each_nd(weights, *weights_initializer);
    for_each_nd(bias, *bias_initializer);

    /*
     * Initialize dw, db
     */
    reset_gradients();
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
     * Save input X, whoch is used in backward()
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
    transform(weights.begin(), weights.end(), bias.begin(), y.begin(),
              [this](const input_vector& w, T bias)
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
     * d_loss/dw = dlos/dy * dy/dw
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
    std::array<std::array<T, num_outputs>, num_inputs> weights_transposed;

    for (size_t i = 0; i < num_inputs; i++)
      {
        for (size_t j = 0; j < num_outputs; j++)
          {
            weights_transposed[i][j] = weights[j][i];
          }
      }

    transform(weights_transposed.begin(), weights_transposed.end(), grad_out.begin(),
              [grad](output_vector& w)
              {
                T val = inner_product(w.begin(), w.end(), grad.begin(), 0.0);
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
    for_each_nd(dw, generate_const<T>());
    for_each_nd(db, generate_const<T>());
  }

};

/*
 * 2D convolution class template
 */
template<int channels_out,      /* output channels */
         int channels_inp,      /* input channels */
         int input_height,      /* input height */
         int input_width,       /* input width */
         int kernel_size = 3,   /* kernel size */
         int stride = 1,        /* stride */
         bool use_bias = false, /* enable bias flag */
         typename T = float,    /* convolution data type */
         void (*weights_initializer)(T&) = random_uniform_initializer<>,
         void (*bias_initializer)(T&) = random_uniform_initializer<> >
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
  static const int output_width  = (input_width  + stride - 1) / stride;
  static const int output_height = (input_height + stride - 1) / stride;
  typedef ndarray<T, channels_out, output_height, output_width> output_array;
  typedef output_array::type conv_output;

  /*
   * Input zero padding required for "same" convolution padding mode.
   * Padding calculation below is consistent with Keras implementation.
   */
  static constexpr int pad_left = ((input_width & 1) == 0) ?  kernel_size / 2 - stride / 2 :  kernel_size / 2;
  static constexpr int pad_right = kernel_size / 2 + kernel_size / 2 - pad_left;

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

  conv_input x;

  /*
   * Default convolution constructor
   */
  Conv2D()
  {
    for_each_nd(weights, *weights_initializer);
    for_each_nd(bias, *bias_initializer);
    for_each_nd(dw, generate_const<T>());
    for_each_nd(db, generate_const<T>());
  }


  /*
   * Forward path computes 2D convolution of input x and kernel weights w
   */
  conv_output forward(const conv_input& input_x)
  {
    conv_output y;

    /*
     * Save input X, used in backward
     */
    x = input_x;

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for_each_nd(y[output_channel], generate_const{use_bias * bias[output_channel]});
      }

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int input_channel = 0; input_channel < channels_inp; input_channel++)
          {
            convolution_nd<T, pad_left, stride, 1>(y[output_channel],
                                                   x[input_channel],
                                                   weights[output_channel][input_channel]);
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
  conv_input backward(const conv_output& grad)
  {
    /*
     * Compute weight gradient dw
     */

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int input_channel = 0; input_channel < channels_inp; input_channel++)
          {
            convolution_nd<T, pad_left, 1, stride>(dw[output_channel][input_channel],
                                                   x[input_channel],
                                                   grad[output_channel]);
          }
      }

    /*
     * Compute bias gradient db
     */
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

    /*
     * Compute output gradient dX
     * dX is convolution of dilated gradient and flipped kernel
     */
    conv_input dx = {};

    for (int output_channel = 0; output_channel < channels_out; output_channel++)
      {
        for (int input_channel = 0; input_channel < channels_inp; input_channel++)
          {
            /*
             * Flip kernel weights
             */
            auto weights_rot180 = weights[output_channel][input_channel];
            rotate_180<T, kernel_size>(weights_rot180);

            /*
             * Dilate graient
             */
            auto grad_dilated = dilate<stride>(grad[output_channel]);

            /*
             * Compute convolution
             */
            convolution_nd<T, pad_right, 1, 1>(dx[input_channel], grad_dilated, weights_rot180);
          }
      }

    return dx;
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
                    auto weight_update =  learning_rate * dw[output_channel][input_channel][i][j];
                    weights[output_channel][input_channel][i][j] -= weight_update;
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
    for_each_nd(dw, generate_const<T>());
    for_each_nd(db, generate_const<T>());
  }

  /*
   * Rotate square 2D array by 180 degrees
   */
  template<typename X, std::size_t N>
  void rotate_180(typename ndarray<X, N, N>::type& x)
  {
    for (std::size_t i = 0; i < N / 2; i++)
      {
        for (std::size_t j = 0; j < N; j++)
          {
            auto t = x[i][j];
            x[i][j] = x[N - i - 1][N - j - 1];
            x[N - i - 1][N - j - 1] = t;
          }
      }

    /*
     * Odd dimension
     */
    if (N & 1)
      {
        for (std::size_t j = 0; j < N / 2; j++)
          {
            auto t = x[N / 2][j];
            x[N / 2][j] = x[N / 2][N - j - 1];
            x[N / 2][N - j - 1] = t;
          }
      }
  }

    template<int dilation>
  auto dilate(const typename ndarray<T, output_height, output_width>::type& x)
  {
    constexpr auto dilated_height = (output_height - 1) * stride  + 1;
    constexpr auto dilated_width = (output_width - 1) * stride + 1;
    typename ndarray<T, dilated_height, dilated_width>::type y = {};

    for (int i = 0; i < output_height; i++)
      {
        for (int j = 0; j < output_width; j++)
          {
            y[i*dilation][j*dilation] = x[i][j];
          }
      }

    return y;
  }

};


/*
 *
 * Rectified Linear Unit (ReLU) class template
 * implements ReLU activarion funtion
 * f(x) = max(x, alpha * x)
 *
 */
template<typename T = float, const T alpha=0.25, std::size_t... Dims>
struct ReLU
{
  typedef ndarray<T, Dims...>::type input_t;
  typedef ndarray<T, Dims...>::type output_t;

  /*
   * x is for saving input to forward() call, used later in the backward() pass.
   */
  input_t x;

  /*
   * ReLU forward pass
   */
  output_t forward(input_t input_x)
  {
    x = input_x;
    for_each_nd(input_x,
                [](T& xi)
                {
                  xi = std::max<T>(xi, alpha * xi);
                });
    return input_x;
  }

  /*
   * ReLU backward pass
   */
  input_t backward(output_t dx)
  {
    for_each_nd(dx, x,
                [](T& dx_i, T& x_i)
                {
                  dx_i = (x_i > 0) ? dx_i : static_cast<T>(alpha * dx_i);
                });

    return dx;
  }

  /*
   * No trainabele weights in ReLU
   */
  void train(float lr)
  {
  }

};

/*
 * Flatten class converts input N-dimentional array to 1 dimentional array
 * in the forward() call, and applies inverse 1-D to N-D conversion in the backward() call
 */
template<typename T = float,
         std::size_t outer_dim = 1,
         std::size_t... Dims>
struct Flatten
{

  typedef ndarray<float, outer_dim, Dims...> input_array;
  typedef input_array::type input_type;
  typedef array<T, input_array::size> output_type;
  static constexpr std::size_t size_inner = {(Dims * ...)};

  output_type forward(const input_type& x)
  {
    output_type y;
    for (size_t i = 0; i < outer_dim; i++)
      {
        for (size_t j = 0; j < size_inner; j++)
          {
            y[i+j*outer_dim] = ndarray<float, Dims...>::ndarray_at(x[i], j);
          }
      }
    return y;
  }

  input_type backward(const output_type& y)
  {
    input_type x;

    for (size_t i = 0; i < outer_dim; i++)
      {
        for (size_t j = 0; j < size_inner; j++)
          {
            ndarray<float, Dims...>::ndarray_at(x[i], j) = y[i+j*outer_dim];
          }
      }

    return x;
  }

  /*
   * No trainable weights in Flatten
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
   * Softmax temperature
   */
  static constexpr T tau = 1.0;
  static constexpr T eps = 2.22044604925e-16;

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
     * Subtract max(X) from X for softmax stability
     */
    T max_x = *std::max_element(input_x.begin(), input_x.end());

    /*
     * compute exp(x_i) / sum(exp(x_i), i=1..N)
     */
    transform(x.begin(), x.end(), y.begin(),
              [&max_x](const T& xi)
              {
                T out = expf( (xi - max_x) / tau);
                return out;
              });

    T sum = accumulate(y.begin(), y.end(), static_cast<T>(0.0)) + eps;
    for_each(y.begin(), y.end(), [sum](T &yi){ yi /= sum;});

    return y;
  }

  /*
   * Softmax backward function
   */
  input_vector backward(const output_vector& grad_inp)
  {
    input_vector grad_out;

    /*
     * Compute Jacobian of Softmax
     */
    std::array<input_vector, num_inputs> J;

    const input_vector y = forward(x);
    int diag_idx = 0;
    transform(y.begin(), y.end(), J.begin(),
              [&diag_idx, y](const auto& y_i)
              {
                auto ret = y;
                for_each(ret.begin(), ret.end(), [y_i](T& y_j){ y_j = -y_i * y_j;});
                ret[diag_idx++] += y_i;
                return ret;
              });

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
        auto ret =  std::get<index>(layers[index]).forward(y_prev);
        return ret;
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
  static constexpr float eps = 2.22044604925e-16;

  /*
   * Forward pass computes CCE loss for inputs y (label) and yhat (predicted)
   */
  static T forward(const input_vector& y, const input_vector& yhat)
  {
    T loss = transform_reduce(y.begin(), y.end(), yhat.begin(), 0.0, plus<T>(),
                              [](const T& y_i, const T& yhat_i)
                              {
                                return y_i * logf(yhat_i + eps);
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
                return -1 * y_i / (yhat_i + eps) + 1.0;
              }
              );

    return de_dy;
  }

};

int main()
{
  const int input_channels = 3;
  const int input_height = 32;
  const int input_width = 32;
  const int kernel_size = 3;
  const int num_classes = 10;
  float learning_rate = 0.001;
  const int print_frequency = 20;
  const int batch_size = 50;
  const int num_epochs = 10;

  /*
   * Accuracy score
   */
  f1<num_classes, float> f1;

  /*
   * DNN input image and label
   */
  ndarray<float, input_channels, input_height, input_width>::type x;
  std::array<float, num_classes> y_true;

  /*
   * Parse CIFAR10 dataset
   */
  cifar<num_classes> train_ds({
      "./cifar-10-batches-bin/data_batch_1.bin",
      "./cifar-10-batches-bin/data_batch_2.bin",
      "./cifar-10-batches-bin/data_batch_3.bin",
      "./cifar-10-batches-bin/data_batch_4.bin",
      "./cifar-10-batches-bin/data_batch_5.bin"
        });
  cifar<num_classes> valid_ds({"./cifar-10-batches-bin/test_batch.bin"});
  cifar_meta<num_classes> meta("./cifar-10-batches-bin/batches.meta.txt");

  printf("traininig dataset: %d records\n", train_ds.num_records);
  printf("validaton dataset: %d records\n", valid_ds.num_records);

  /*
   * DNN contains several downsampling convolution layers followed by fully connected layer.
   */
  Sequential< Conv2D<8,                      /* output channels */
                     input_channels,         /* input channels */
                     input_height,           /* input height */
                     input_width,            /* input width */
                     kernel_size,            /* convolution kernel size */
                     1,                      /* stride */
                     false>,                 /* use bias */

              ReLU<float, 0.25f, 8, input_height, input_width>,

              Conv2D<8,
                     8,
                     input_height,
                     input_width,
                     kernel_size,
                     2,
                     false>,

              ReLU<float, 0.25f, 8, input_height / 2, input_width / 2>,

              Conv2D<16,
                     8,
                     input_height / 2,
                     input_width / 2,
                     kernel_size,
                     1,
                     false>,

              ReLU<float, 0.25f, 16, input_height / 2, input_width / 2>,

              Conv2D<16,
                     16,
                     input_height / 2,
                     input_width / 2,
                     kernel_size,
                     2,
                     false>,

              ReLU<float, 0.25f, 16, input_height / 4, input_width / 4>,

              Conv2D<32,
                     16,
                     input_height / 4,
                     input_width / 4,
                     kernel_size,
                     1,
                     false>,

              ReLU<float, 0.25f, 32, input_height / 4, input_width / 4>,

              Conv2D<32,
                     32,
                     input_height / 4,
                     input_width / 4,
                     kernel_size,
                     2,
                     false>,

              ReLU<float, 0.25f, 32, input_height / 8, input_width / 8>,

              Flatten<float, 32, input_height / 8, input_width / 8>,

              Dense<32 * input_height  * input_width / (8 * 8), num_classes>,

              Softmax<num_classes>> net;

  /*
   * Loss function
   */
  CCE<num_classes> cce;

  /*
   * Training loop
   */
  for (auto epoch = 0; epoch < num_epochs; epoch++)
    {
      float loss_train_avg = 0.0;
      float loss_valid_avg = 0.0;
      float loss = 0;

      train_ds.rewind();

      const int num_iter = train_ds.num_records / batch_size;
      for (auto iter = 0; iter < num_iter; iter++)
        {
          for (auto idx = 0; idx < batch_size; idx++)
            {
              /*
               * Read next image and label
               */
              auto ret = train_ds.read_next(x, y_true);
              assert(ret == 0);

              /*
               * Forward path
               */
              auto y_pred = net.forward(x);

              /*
               * Compute loss
               */
              loss = cce.forward(y_true, y_pred);
              assert(false == std::isnan(loss));
              loss_train_avg += loss;

              /*
               * Backward path
               */
              auto dy = cce.backward(y_true, y_pred);
              net.backward(dy);

              /*
               * Update weights
               */
              net.train(learning_rate);

            }

          /*
           * Print stats
           */
          if ( (iter % print_frequency) == 0)
            {
              printf("epoch %d/%d; iter %d/%d; avg. training loss: %7.4f\r",
                     epoch+1, num_epochs, iter, num_iter, loss_train_avg / ((iter+1) * batch_size));
              fflush(stdout);
            }

        }
      printf("epoch %d/%d; iter %d/%d; avg. training loss: %7.4f\n",
             epoch+1, num_epochs, num_iter, num_iter, loss_train_avg / (num_iter * batch_size));

      loss_train_avg /= train_ds.num_records;

      /*
       * Validation loop
       */
      valid_ds.rewind();
      f1.reset();

      for (auto iter = 0; iter < valid_ds.num_records; iter++)
        {
          auto ret = valid_ds.read_next(x, y_true);
          assert(ret == 0);

          auto y_pred = net.forward(x);
          auto loss = cce.forward(y_true, y_pred);

          loss_valid_avg += loss;
          f1.update(y_true, y_pred);
        }
      loss_valid_avg /= valid_ds.num_records;

      printf("epoch %d/%d; avg. train loss: %7.4f; avg. valid loss: %7.4f; f1: %5.3f\n",
             epoch+1, num_epochs, loss_train_avg, loss_valid_avg, f1.score());
    }

  return 0;
}
