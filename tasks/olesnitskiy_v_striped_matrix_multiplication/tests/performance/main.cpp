#include <gtest/gtest.h>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "olesnitskiy_v_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"
#include "olesnitskiy_v_striped_matrix_multiplication/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {

class OlesnitskiyVStripedMatrixMultiplicationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    const size_t size = 1024;
    
    size_t rows_A = size;
    size_t cols_A = size;
    size_t rows_B = size;
    size_t cols_B = size;
    
    size_t total_A = rows_A * cols_A;
    size_t total_B = rows_B * cols_B;
    
    std::vector<double> matrix_A(total_A);
    std::vector<double> matrix_B(total_B);
    for (size_t i = 0; i < total_A; ++i) {
      matrix_A[i] = static_cast<double>((i * 37) % 1000) / 1000.0;
    }
    for (size_t i = 0; i < total_B; ++i) {
      matrix_B[i] = static_cast<double>((i * 73 + 50) % 1000) / 1000.0;
    }
    input_data_ = std::make_tuple(rows_A, cols_A, matrix_A, rows_B, cols_B, matrix_B);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto& [out_rows, out_cols, out_data] = output_data;
    if (out_data.empty() || out_rows != 1024 || out_cols != 1024) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OlesnitskiyVStripedMatrixMultiplicationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}
const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OlesnitskiyVStripedMatrixMultiplicationMPI, OlesnitskiyVStripedMatrixMultiplicationSEQ>(PPC_SETTINGS_olesnitskiy_v_striped_matrix_multiplication);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = OlesnitskiyVStripedMatrixMultiplicationPerfTests::CustomPerfTestName;
INSTANTIATE_TEST_SUITE_P(RunModeTests, OlesnitskiyVStripedMatrixMultiplicationPerfTests, kGtestValues, kPerfTestName);
}  // namespace olesnitskiy_v_striped_matrix_multiplication