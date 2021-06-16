#pragma once

#include <array>
#include <algorithm>

template<size_t num_classes=10, typename T=float>
struct f1
{
  std::array<int, num_classes> fp;
  std::array<int, num_classes> fn;
  std::array<int, num_classes> tp;
  static constexpr float eps = 0.000001;

  f1()
  {
    reset();
  }

  void reset()
  {
    fp.fill({});
    fn.fill({});
    tp.fill({});
  }

  void update(const std::array<T, num_classes>& y_true, const std::array<T, num_classes>& y_pred)
  {
    auto idx_true = std::distance(y_true.begin(), std::max_element(y_true.begin(), y_true.end()));
    auto idx_pred = std::distance(y_pred.begin(), std::max_element(y_pred.begin(), y_pred.end()));
    fp[idx_pred] += 1;
    fn[idx_true] += 1;
    if (idx_true == idx_pred)
      {
        tp[idx_true] += 1;
      }
  }

  float score()
  {

    std::array<float, num_classes> precision;
    std::array<float, num_classes> recall;
    std::array<float, num_classes> scores;

    std::transform(tp.begin(), tp.end(), fp.begin(), precision.begin(),
                   [](const int& tp_i, const int& fp_i)
                   {
                     return static_cast<float>(tp_i) / static_cast<float>(fp_i + eps);
                   });
    std::transform(tp.begin(), tp.end(), fn.begin(), recall.begin(),
                   [](const int& tp_i, const int& fn_i)
                   {
                     return static_cast<float>(tp_i) / static_cast<float>(fn_i + eps);
                   });
    std::transform(precision.begin(), precision.end(), recall.begin(), scores.begin(),
                   [](const float& precision_i, const float& recall_i)
                   {
                     auto score =  2.0 * precision_i * recall_i / (precision_i + recall_i + eps);
                     return score;
                   });

    constexpr auto print_fn = [](const float& x)  -> void {printf("%.2f ", x);};
    //printf("fp=");
    //std::for_each(fp.begin(), fp.end(), print_fn);
    //printf("\n");

    auto score_total = std::accumulate(scores.begin(), scores.end(), static_cast<float>(0.0));
    return score_total / static_cast<float>(num_classes);
  }
};