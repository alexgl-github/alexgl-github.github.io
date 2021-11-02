#pragma once

#include <array>
#include <numeric>

/*
 * N-dimentional array definition
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
 * terminating N-dimentional array recursive ndarray<>
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
 * 0-dimentional inner product terminating N-dimentional definition below
 */
template<typename T, int dilation>
static T inner_product_nd(const T& x,
                          const T& w)
{
  return x * w;
}

/*
 * Recursive definition of inner product of N dimentional inputs x and w
 */
template<typename T,
         int dilation,
         typename type_x,
         typename type_w,
         typename... Idx>
static T inner_product_nd(const type_x& x,
                              const type_w& w,
                              const int idx_outer,
                              Idx... idx_inner)
{
  int x_size = static_cast<int>(x.size());
  int w_size = (static_cast<int>(w.size()) - 1) * dilation + 1;
  const int pad_left  = (idx_outer < 0) ? (- idx_outer) : 0;
  const int pad_right =
    (idx_outer > x_size - w_size) ?
    (idx_outer - x_size + w_size) : 0;

  T sum = 0;
  for (ssize_t k = pad_left; k < (static_cast<int>(w.size()) - pad_right); k++)
    {
      sum += inner_product_nd<T, dilation>(x[idx_outer + k * dilation], w[k], idx_inner...);
    }

  return sum;
};

/*
 * Recursive definition of convolutiuon of N dimentional inputs x and w
 *  pad_size: input x will be padded by pad_size zeros in each dimention
 *  diltion: kernel w will be dilated by dilation zeros
 */
template<typename T,
         int pad_size,
         int stride,
         int dilation,
         typename X,
         typename W,
         typename... Idx>
void convolution_nd(T & y,
                    const X & x,
                    const W & w,
                    Idx... idx)
{
  y += inner_product_nd<T, dilation>(x, w, idx...);
}

template<typename T,
         int pad_size,
         int stride,
         int dilation,
         typename Y,
         std::size_t N,
         typename X,
         typename W,
         typename... Idx>
void convolution_nd(std::array<Y, N> & y,
                 const X & x,
                 const W & w,
                 Idx... idx)
{
  for (int i = 0; i < static_cast<int>(N); i++)
    {
      convolution_nd<T, pad_size, stride, dilation>(y[i], x, w, idx..., i * stride - pad_size);
    }
}


/*
 * Apply function for each item in N-dimentional container
 */
template<typename T, typename F>
auto for_each_nd(T& x, F func)
{
  func(x);
}

template<typename T, std::size_t N, typename F>
auto for_each_nd(std::array<T, N>& x, F func)
{
  std::for_each(x.begin(), x.end(), [func](auto& x_i)
                {
                  for_each_nd(x_i, func);
                });
}

template<typename T, typename F>
auto for_each_nd(T& x, T& y, F func)
{
  func(x, y);
}

template<typename T, std::size_t N, typename F>
auto for_each_nd(std::array<T, N>& x, std::array<T, N>& y, F func)
{
  auto first1 = x.begin();
  auto last1 = x.end();
  auto first2 = y.begin();
  for (; first1 != last1; ++first1, ++first2)
    {
      for_each_nd(*first1, *first2, func);
    }
}

/*
 * N-dimentional array print function
 */
template<typename T>
auto print_n(const T& x, const char* fmt)
{
  printf(fmt, x);
}

template<typename T, size_t N>
  auto print_n(const std::array<T, N>& x, const char* fmt="%.2f, ")
{
  printf("[");
  std::for_each(x.begin(), x.end(),
                [fmt](const auto& xi)
                {
                  print_n(xi, fmt);
                });
  printf("], \n");
}
