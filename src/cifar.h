#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include "ndarray.h"

/*
 * CIFAR-10 dataset
 *
 * Dataset description and format can be found at https://www.cs.toronto.edu/~kriz/cifar.html
 *
 * Dataset contains 5 training binary files and 1 validation binary:
 *  data_batch_1.bin
 *  data_batch_2.bin
 *  data_batch_3.bin
 *  data_batch_4.bin
 *  data_batch_5.bin
 *  test_batch.bin
 *
 * Dataset binary files are formatted as follows:
 *
 * <1 x label><3072 x pixel>
 * <1 x label><3072 x pixel>
 * ...
 * <1 x label><3072 x pixel>
 *
 * The first byte is image label, the next 3072 bytes are pixel values.
 * The first 1024 bytes are the red channel values, the next 1024 the green channel, and the final 1024 the blue channel.
 * The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
 *
 */

/*
 * Default scaling funtion to convert from uint8 pixels in [0:255] range to floating point values in [0, 1) range
 */
auto scale = [](float& ro, float& go, float& bo, uint8_t r, uint8_t g, uint8_t b) -> void
{
  ro = (static_cast<float>(r)) / 256.0;
  go = (static_cast<float>(g)) / 256.0;
  bo = (static_cast<float>(b)) / 256.0;
};

/*
 * CIFAR dataset class
 *
 * num_classes=10 and image size=28x28 are as per dataset specifiction
 *
 * typename T=float is type of the data returned by read_next_image call
 *
 */
template<int num_classes = 10,
         int height      = 32,
         int width       = 32,
         int channels    = 3,
         typename T      = float,
         void (*transform)(T&, T&, T&, uint8_t, uint8_t, uint8_t) = scale >
struct cifar
{
  enum error
    {
      CIFAR_OK = 0,
      CIFAR_EOF = -1,
      CIFAR_ERROR = -2
    };

  /* images file descriptor */
  size_t batch_index = -1;
  std::vector<int> fds;
  int num_records = 0;
  static constexpr int record_size_bytes = height * width * channels + 1;

  /*
   * Initialize dataset from batch files
   */
  cifar(std::vector<const char*> batches)
  {
    for (auto batch_path: batches)
      {
        int fd = open(batch_path, O_RDONLY);
        assert(fd != -1);
        auto size = file_size(fd);
        num_records += size / record_size_bytes;
        fds.push_back(fd);
      }
    batch_index = 0;
  }

  /*
   * Close file descriptors in destructor
   */
  ~cifar()
  {
    for (auto fd: fds)
      {
        assert(fd != -1);
        close(fd);
      }
  }

  /*
   * Read next image and label
   * Apply transformation to image pixes as defined in template parameters
   * Convert label to on-hot encoded format
   */
  int read_next(ndarray<T, channels, height, width>::type& x,
               std::array<T, num_classes>& label_onehot)
  {
    int ret;
    typename ndarray<uint8_t, channels, height, width>::type raw;
    uint8_t label = -1;

    ret = read_from_batch(&label, sizeof(label));
    if (ret != sizeof(label))
      {
        return CIFAR_EOF;
      }

    assert(label < num_classes);

    std::fill(label_onehot.begin(), label_onehot.end(), 0);
    label_onehot[label] = 1.0;

    for(int channel = 0; channel < channels; channel++)
      {
        for(int i = 0; i < height; i++)
          {
            ret = read_from_batch(raw[channel][i].data(), width);
            if (ret != width)
              {
                return CIFAR_EOF;
              }
          }
      }

    for(int i = 0; i < height; i++)
      {
        for(int j = 0; j < width; j++)
          {
            (*transform)(x[0][i][j], x[1][i][j], x[2][i][j], raw[0][i][j], raw[1][i][j], raw[2][i][j]);
          }
      }

    return CIFAR_OK;;
  }

  /*
   * Rewind dataset to start
   */
  void rewind()
  {
    for (auto fd: fds)
      {
        assert(fd != -1);
        lseek(fd, 0, SEEK_SET);
      }
    batch_index = 0;
  }

  /*
   * Read binary data from current batch.
   * Go to the next batch if at the end of current batch.
   * Return 0 if at the end of the last batch.
   */
  int read_from_batch(uint8_t* data, size_t size)
  {
    int ret = read(fds[batch_index], data, size);

    if (ret == 0)
      {
        batch_index += 1;
        if (batch_index >= fds.size())
          {
            return ret;
          }
        ret = read(fds[batch_index], data, size);
      }

    return ret;
  }

  /*
   * Return file size for file descriptor fd
   */
  static size_t file_size(int fd)
  {
    auto pos = lseek(fd, (size_t)0, SEEK_CUR);
    auto size = lseek(fd, (size_t)0, SEEK_END);
    lseek(fd, pos, SEEK_SET);
    return size;
  }

};


/*
 *
 * Using batches.meta.txt, build map of integer label to string
 *
 */
template<int num_classes = 10>
struct cifar_meta
{
  std::unordered_map<int, std::string> labels;

  cifar_meta(const char* path_meta="batches.meta.txt")
  {
    const char* ws = " \t\n\r\f\v";
    const int max_label_size = 255;
    char label_str[max_label_size + 1];
    FILE* fd_meta = fopen(path_meta, "r");
    assert(fd_meta != NULL);
    int label_idx = 0;
    while(fgets(label_str, max_label_size, fd_meta) != NULL)
      {
        std::string label = std::string(label_str);
        label.erase(label.find_last_not_of(ws) + 1);
        if (label.empty())
          {
            continue;
          }
        labels[label_idx] = label;
        label_idx++;
        printf("label %d %s\n", label_idx-1, label.c_str());
      }
    fclose(fd_meta);
    assert(labels.size() == num_classes);
  }

  const char* operator[](int label_idx)
  {
    std::string str =  labels[label_idx];
    return str.c_str();
  }
};

