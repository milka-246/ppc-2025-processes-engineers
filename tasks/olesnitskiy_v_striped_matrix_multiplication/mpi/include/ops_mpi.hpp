#pragma once

#include <mpi.h>
#include <vector>
#include <tuple>

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
  static std::vector<int> calculate_counts(int total, int num_parts);
  static std::vector<int> calculate_displacements(const std::vector<int>& counts);

 private:
  int rank_;
  int world_size_;
  
  size_t rows_A_;
  size_t cols_A_;
  std::vector<double> data_A_;
  
  size_t rows_B_;
  size_t cols_B_;
  std::vector<double> data_B_;
  
  size_t rows_C_;
  size_t cols_C_;
  std::vector<double> result_C_;
};

}  // namespace olesnitskiy_v_striped_matrix_multiplication
