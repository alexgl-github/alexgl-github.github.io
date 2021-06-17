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
  return 2 * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0;
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
  vector<input_vector> weights;

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
  vector<array<T*, num_outputs>> weights_transposed;

  /*
   * dw is accumulating weight updates in backward() pass.
   */
  vector<input_vector> dw;

  /*
   * db is accumulating bias updates in backward() pass.
   */
  output_vector db;

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
    dw.resize(num_outputs);
    reset_grads();
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
  input_vector backward(const input_vector& input, const output_vector grad)
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
              [input](input_vector& left, const T& grad_i)
              {
                /* compute outer product for each row */
                auto row = input;
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

    reset_grads();
  }

  /*
   * Reset weigth and bias gradient accumulators
   */
  void reset_grads()
  {
    for (input_vector& dw_i: dw)
      {
        generate(dw_i.begin(), dw_i.end(), const_initializer<const_zero>);
      }
    generate(db.begin(), db.end(), const_initializer<const_zero>);
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

  /*
   * Sigmoid forward pass
   */
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

  /*
   * Sigmoid backward pass
   */
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

    T sum = accumulate(y.begin(), y.end(), static_cast<T>(0.0));
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


int main(void)
{
  /*
   * Path to MNIST train and validation data
   */
  const char* train_images_path = "./data/train-images-idx3-ubyte";
  const char* train_labels_path = "./data/train-labels-idx1-ubyte";
  const char* validation_images_path = "./data/t10k-images-idx3-ubyte";
  const char* validation_labels_path = "./data/t10k-labels-idx1-ubyte";

  /*
   * Number of classes in MNIST dataset
   */
  const int num_classes = 10;

  /*
   * MNIST image dimentions
   */
  const int num_rows = 28;
  const int num_cols = 28;
  const int image_size = num_rows * num_cols;

  /*
   * Train for 5 epochs
   */
  const int num_epochs = 5;

  /*
   * Print report every 10000 iterations
   */
  const int report_interval = 10000;

  /*
   * Storage for next image and label
   */
  std::array<float, image_size> image;
  std::array<float, num_classes> label;

  /*
   * Training learning rate and batch size
   */
  float learning_rate = 0.01;
  int batch_size = 100;

  /*
   * MNIST train dataset
   */
  mnist train_dataset(train_images_path, train_labels_path);

  /*
   * MNIST validation dataset
   */
  mnist validation_dataset(validation_images_path, validation_labels_path);


  printf("train dataset: %d images and %d labels\n", train_dataset.number_of_images, train_dataset.number_of_labels);
  printf("validation dataset: %d images and %d labels\n", validation_dataset.number_of_images, validation_dataset.number_of_labels);

  /*
   * Number of iterations in epoch is number of records in the dataset
   */
  const auto iterations = train_dataset.number_of_images;

  /*
   * Create DNN layers and the loss
   */
  Dense<image_size, 128> dense1;
  Sigmoid<128> sigmoid1;
  Dense<128, num_classes> dense2;
  Softmax<num_classes> softmax;
  CCE<num_classes> loss_fn;
  f1<num_classes, float> f1_train;
  f1<num_classes, float> f1_validation;

  /*
   * shortcut for mnust error code
   */
  using mnist_error = mnist<10, 28*28, float>::error;

  /*
   * Training loop
   * Train for num_epochs
   */
  for (auto epoch=0; epoch < num_epochs; epoch++)
    {
      /*
       * Reset dataset positons, loss accuracy counters, and clocks
       */
      train_dataset.rewind();
      float loss_epoch = 0;
      f1_train.reset();
      auto ts = high_resolution_clock::now();

      /*
       * Train for number of iterations per each epoch
       */
      for (auto iter = 0; iter < iterations / batch_size; iter++)
        {

          /*
           * Repeat forward path for batch_size
           */
          for (auto batch = 0; batch < batch_size; batch++)
            {

              auto ret = train_dataset.read_next(image, label);
              if (ret == mnist_error::MNIST_EOF)
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
               * Update f1 score
               */
              f1_train.update(label, y4);

              /*
               * Back propagate loss
               */
              auto dloss_dy4 = loss_fn.backward(label, y4);
              auto dy4_dy3 = softmax.backward(y4, dloss_dy4);
              auto dy3_dy2 = dense2.backward(y2, dy4_dy3);
              auto dy2_dy1 = sigmoid1.backward(y2, dy3_dy2);
              dense1.backward(image, dy2_dy1);

              /*
               * Accumulate loss for reporting
               */
              loss_epoch += loss;
            }

          /*
           * Update dense layers weights once per batch
           */
          dense2.train(learning_rate);
          dense1.train(learning_rate);

          /*
           * Print loss and accuracy per each reporting interval
           */
          if (epoch == 0 && iter == 0)
            {
              printf("epoch=%d/%d iteration=%d/%d loss=%.5f f1=%.5f\n",
                     epoch,
                     num_epochs,
                     (iter+1), iterations/batch_size,
                     loss_epoch / ((iter + 1)*batch_size),
                     f1_train.score());
            }
          if ( (((iter+1) * batch_size) % report_interval) == 0)
            {
              printf("epoch=%d/%d iteration=%d/%d loss=%.5f f1=%.5f\n",
                     epoch,
                     num_epochs,
                     (iter+1), iterations/batch_size,
                     loss_epoch / ((iter + 1)*batch_size),
                     f1_train.score());
            }
        }

      /*
       * Capture run time per epoch
       */
      auto te = high_resolution_clock::now();
      auto dt = (float)duration_cast<seconds>(te - ts).count();

      /*
       * Average loss per epoch
       */
      loss_epoch = loss_epoch / iterations;

      /*
       * Print epoch stats
       */
      printf("epoch %d/%d time/epoch=%.5f sec; time left=%.4f hr; avg loss=%.5f; f1=%.5f\n",
             epoch,
             num_epochs,
             dt, dt * (num_epochs - epoch - 1) / (60*60),
             loss_epoch,
             f1_train.score());


      /*
       * Validation loop
       */
      float loss_validation = 0;
      validation_dataset.rewind();
      f1_validation.reset();

      /*
       * Repeat each epoch for entire validation dataset
       */
      for (auto iter = 0; iter < validation_dataset.number_of_images; iter++)
        {
          /*
           * Read next image and label
           */
          auto ret = validation_dataset.read_next(image, label);
          if (ret == mnist_error::MNIST_EOF)
            {
              break;
            }

          /*
           * Forward path
           */
          auto y1 = dense1.forward(image);
          auto y2 = sigmoid1.forward(y1);
          auto y3 = dense2.forward(y2);
          auto y4 = softmax.forward(y3);
          auto loss = loss_fn.forward(label, y4);

          /*
           * Update validation loss and counters
           */
          loss_validation += loss;
          f1_validation.update(label, y4);
        }

      /*
       * Report validation loss and f1 score
       */
      loss_validation = loss_validation / validation_dataset.number_of_images;
      printf("epoch %d/%d validation loss: %f  f1: %.5f\n",
             epoch, num_epochs, loss_validation, f1_validation.score());
    }

  return 0;
}
