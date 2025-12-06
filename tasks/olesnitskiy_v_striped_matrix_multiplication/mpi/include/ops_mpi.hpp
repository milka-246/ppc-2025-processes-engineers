#pragma once

#include <mpi.h>

#include <cmath>
#include <cstddef>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "util/include/util.hpp"

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
  bool RunOnSingleProcess();
  static std::vector<int> CalculateCounts(int total, int num_parts);
  static std::vector<int> CalculateDisplacements(const std::vector<int> &counts);

 private:
  int rank_;
  int world_size_;

  size_t rows_a_;
  size_t cols_a_;
  std::vector<double> data_a_;

  size_t rows_b_;
  size_t cols_b_;
  std::vector<double> data_b_;

  size_t rows_c_;
  size_t cols_c_;
  std::vector<double> result_c_;
};

}  // namespace olesnitskiy_v_striped_matrix_multiplication
