#pragma once

#include <cstddef>
#include <vector>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {

class OlesnitskiyVStripedMatrixMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OlesnitskiyVStripedMatrixMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool MultiplySimple();
  bool MultiplyStriped();

  [[nodiscard]] int FindCommonDivisor(int a, int b, int max_divisor) const;

  size_t rows_a_;
  size_t cols_a_;
  std::vector<double> data_a_;
  size_t rows_b_;
  size_t cols_b_;
  std::vector<double> data_b_;
  size_t rows_c_;
  size_t cols_c_;
  std::vector<double> result_c_;
  int num_stripes_;
};
}  // namespace olesnitskiy_v_striped_matrix_multiplication
