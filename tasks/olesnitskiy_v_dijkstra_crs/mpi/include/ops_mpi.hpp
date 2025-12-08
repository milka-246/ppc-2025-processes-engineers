#pragma once

#include <vector>

#include "olesnitskiy_v_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace olesnitskiy_v_dijkstra_crs {
class OlesnitskiyVDijkstraCrsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit OlesnitskiyVDijkstraCrsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> DijkstraCrsMpi(const GraphCRS &graph);
};
}  // namespace olesnitskiy_v_dijkstra_crs
