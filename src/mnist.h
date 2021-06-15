#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <cassert>

/*
 * Full dataset format specification is at http://yann.lecun.com/exdb/mnist/
 *
 * TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 * 0004     32 bit integer  60000            number of items
 * 0008     unsigned byte   ??               label
 * 0009     unsigned byte   ??               label
 * ........
 * xxxx     unsigned byte   ??               label
 * The labels values are 0 to 9.
 *
 * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel
 * 0017     unsigned byte   ??               pixel
 * ........
 * xxxx     unsigned byte   ??               pixel
 *
 * Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
 *
 */

/*
 * MNIST dataset class
 *
 * num_classe=10 and image size=28x28 are as per dataset specifiction
 *
 * typename T=float is type of the data returned by read_next_image call
 *
 */
template<size_t num_classes=10, size_t image_size=28*28, typename T=float>
struct mnist
{
  enum error
    {
      MNIST_OK = 0,
      MNIST_EOF = -1,
      MNIST_ERROR = -2
    };

  /* images file descriptor */
  int fd_images = -1;
  /* labels file descriptor */
  int fd_labels = -1;
  /* images magic number */
  const uint32_t magic_image_value = 0x00000803;
  /* labels magic number */
  const uint32_t magic_label_value = 0x00000801;
  /* number of images in the dataset */
  int number_of_images = 0;
  /* number of labels in the dataset */
  int number_of_labels = 0;

  /*
   * Initialize MNIST dataset from image file and labels file
   * Read and verify magic numbers
   * Verify number if images matches number of labels
   * Verify read image size matches template parameter image_size
   */
  mnist(const char* path_images, const char* path_labels)
  {
    uint32_t magic_image,  magic_label;
    int rows, cols;

    /*
     * Read images header
     */
    fd_images = open(path_images, O_RDONLY);
    assert(fd_images != -1);

    read(fd_images, &magic_image, sizeof(magic_image));
    assert(endian_swap<uint32_t>(magic_image) == magic_image_value);

    read(fd_images, &number_of_images, sizeof(number_of_images));
    number_of_images = endian_swap<int>(number_of_images);

    read(fd_images, &rows, sizeof(rows));
    read(fd_images, &cols, sizeof(cols));
    rows = endian_swap<int>(rows);
    cols = endian_swap<int>(cols);
    assert(rows * cols == image_size);

    /*
     * Read labels header
     */
    fd_labels = open(path_labels, O_RDONLY);
    assert(fd_labels != -1);

    read(fd_labels, &magic_label, sizeof(magic_label));
    assert(endian_swap<uint32_t>(magic_label) == magic_label_value);

    read(fd_labels, &number_of_labels, sizeof(number_of_labels));
    number_of_labels = endian_swap<int>(number_of_labels);
    assert(number_of_images == number_of_labels);
  }

  /*
   * Close file descriptors in dtor
   */
  ~mnist()
  {
    close(fd_images);
    close(fd_labels);
  }

  /*
   * Read image pixel values
   */
  int read_next_image(std::array<T, image_size>& data)
  {
    /*
     * Read uint_8 pixel values
     */
    std::array<uint8_t, image_size> raw_data;
    ssize_t ret = read(fd_images, raw_data.data(), image_size);
    if (ret == 0)
      {
        return MNIST_EOF;
      }

    assert(ret == image_size);

    /*
     * convert to floating point and normalize
     */
    std::transform(raw_data.begin(), raw_data.end(), data.begin(),
              [](const uint8_t x) {
                return static_cast<T>(x) / 256.0;
              }
              );
    return MNIST_OK;
  }

  /*
   * Read image label and return on one-hot encoded formt
   */
  int read_next_label(std::array<T, num_classes>& label)
  {
    uint8_t label_index = 0;

    /*
     * Fill one hot array with zeros
     */
    std::fill(label.begin(), label.end(), 0);

    /*
     * Read label value
     */
    ssize_t ret = read(fd_labels, &label_index, sizeof(label_index));

    if (ret == 0)
      {
        return MNIST_EOF;
      }

    assert(ret == sizeof(label_index));
    assert(label_index < num_classes);

    /*
     * Set one hot array at label value index to 1
     */
    label[label_index] = 1.0;
    return MNIST_OK;
  }

  /*
   * Wrapper for read_next_image() and read_next_label()
   */
  int read_next(std::array<T, image_size>& image, std::array<T, num_classes>& label)
  {
    auto ret1 = read_next_image(image);
    auto ret2 = read_next_label(label);
    if (ret1 == MNIST_EOF || ret2 == MNIST_EOF)
      {
        return MNIST_EOF;
      }
    assert(ret1 == MNIST_OK);
    assert(ret2 == MNIST_OK);
    return MNIST_OK;
  }

  /*
   * Reset file offsets to the beginning of data
   * skipping headers
   */
  void rewind()
  {
    if (fd_images != -1 && fd_labels != -1)
      {
        /*
         * seek to data offsets in labels and images. For offsets see start of mnist.h
         */
        lseek(fd_images, 16, SEEK_SET);
        lseek(fd_labels, 8, SEEK_SET);
      }
  }

  /*
   * Utility to convert betwwen big/little endiannes
   */
  template<typename S = uint32_t>
  static S endian_swap(T val)
  {
    typedef union
    {
      S val;
      unsigned char array[sizeof(S)];
    } swap_t;
    swap_t src, dst;
    src.val = val;
    for (size_t i = 0; i < sizeof(S); i++)
      {
        dst.array[i] = src.array[sizeof(S) - i - 1];
      }
    return dst.val;
  }

};
