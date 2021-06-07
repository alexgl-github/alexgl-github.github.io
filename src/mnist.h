#pragma once
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>

template<size_t num_classes=10, size_t image_size=28*28, typename T=float>
struct mnist
{
  int fd_images = -1;
  int fd_labels = -1;
  const uint32_t magic_image_value = 0x00000803;
  const uint32_t magic_label_value = 0x00000801;
  int number_of_images = 0;
  int number_of_labels = 0;

  mnist(const char* path_images, const char* path_labels)
  {
    uint32_t magic_image,  magic_label;
    int rows, cols;

    fd_images = open(path_images, O_RDONLY);

    if (fd_images == -1 ||
        sizeof(magic_image) != read(fd_images, &magic_image, sizeof(magic_image)) ||
        endian_swap<uint32_t>(magic_image) != magic_image_value )
      {
        printf("error reading %s\n", path_images);
        return;
      }

    read(fd_images, &number_of_images, sizeof(number_of_images));
    read(fd_images, &rows, sizeof(rows));
    read(fd_images, &cols, sizeof(cols));
    number_of_images = endian_swap<int>(number_of_images);
    rows = endian_swap<int>(rows);
    cols = endian_swap<int>(cols);
    printf("magic_image=%x count=%d %dx%d\n", magic_image, number_of_images, rows, cols);

    fd_labels = open(path_labels, O_RDONLY);

    if (fd_labels == -1 ||
        sizeof(magic_label) != read(fd_labels, &magic_label, sizeof(magic_label)) ||
        endian_swap<uint32_t>(magic_label) != magic_label_value )
      {
        printf("error reading %s\n", path_labels);
        return;
      }

    read(fd_labels, &number_of_labels, sizeof(number_of_labels));
    number_of_labels = endian_swap<int>(number_of_labels);

    printf("magic_label=%x count=%d\n", magic_label, number_of_labels);

    if (number_of_images != number_of_labels)
      {
        printf("error, number of images should equal number of labels\n");
        return;
      }

    if (rows * cols != image_size)
      {
        printf("error, number of images should equal number of labels\n");
        return;
      }
  }

  ~mnist()
  {
    close(fd_images);
    close(fd_labels);
  }

  int read_next_image(std::array<T, image_size>& data , bool normalize=true)
  {
    uint8_t val;
    for (size_t i = 0; i < image_size; i++)
      {
        if (sizeof(val) != read(fd_images, &val, sizeof(val)))
          {
            return -1;
          }
        data[i] = static_cast<T>(val);
        if (normalize)
          {
            data[i] = data[i] / 256.0;
          }
      }
    return image_size;
  }

  int read_next_label(std::array<T, num_classes>& label)
  {
    uint8_t label_index;

    std::fill(label.begin(), label.end(), 0);
    if (sizeof(label_index) != read(fd_labels, &label_index, sizeof(label_index)))
      {
        return -1;
      }

    label[label_index] = 1.0;
    return 1;
  }

  void rewind()
  {
    if (fd_images != -1 && fd_labels != -1)
      {
        lseek(fd_images, 0, SEEK_SET);
        lseek(fd_labels, 0, SEEK_SET);
      }
  }

  template<typename S = uint32_t>
  static S endian_swap(T val)
  {
    typedef union
    {
      S val;
      unsigned char array[sizeof(S)];
    } swap_t;
    swap_t src;
    swap_t dst;
    src.val = val;
    for (size_t i = 0; i < sizeof(S); i++)
      {
        dst.array[i] = src.array[sizeof(S) - i - 1];
      }
    return dst.val;
  }

};
