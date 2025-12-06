#pragma once

#include <cstddef>
#include <vector>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {

class OlesnitskiyVStripedMatrixMultiplicationMPI : public ppc::task::Task<InType, OutType> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit OlesnitskiyVStripedMatrixMultiplicationMPI(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<int> CalculateCounts(int total, int num_parts);
  static std::vector<int> CalculateDisplacements(const std::vector<int> &counts);
  bool RunOnSingleProcess();

  size_t rows_a_{0};
  size_t cols_a_{0};
  std::vector<double> data_a_;

  size_t rows_b_{0};
  size_t cols_b_{0};
  std::vector<double> data_b_;

  size_t rows_c_{0};
  size_t cols_c_{0};
  std::vector<double> result_c_;

  int rank_{-1};
  int world_size_{-1};
};

}  // namespace olesnitskiy_v_striped_matrix_multiplication
