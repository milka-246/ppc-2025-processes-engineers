#pragma once

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

  int find_common_divisor(int a, int b, int max_divisor) const;
  
  size_t rows_A_;
  size_t cols_A_;
  std::vector<double> data_A_;
  size_t rows_B_;
  size_t cols_B_;
  std::vector<double> data_B_;
  size_t rows_C_;
  size_t cols_C_;
  std::vector<double> result_C_;
  int num_stripes_;
};
}  // namespace olesnitskiy_v_striped_matrix_multiplication