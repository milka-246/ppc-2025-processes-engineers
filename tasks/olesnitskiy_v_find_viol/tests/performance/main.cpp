#include <gtest/gtest.h>

#include "olesnitskiy_v_find_viol/common/include/common.hpp"
#include "olesnitskiy_v_find_viol/mpi/include/ops_mpi.hpp"
#include "olesnitskiy_v_find_viol/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace olesnitskiy_v_find_viol {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OlesnitskiyVFindViolMPI, OlesnitskiyVFindViolSEQ>(PPC_SETTINGS_olesnitskiy_v_find_viol);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace olesnitskiy_v_find_viol
