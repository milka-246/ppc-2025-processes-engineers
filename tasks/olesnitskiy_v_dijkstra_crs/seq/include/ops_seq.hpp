#pragma once
#include "olesnitskiy_v_dijkstra_crs/common/include/common.hpp"
#include "task/include/task.hpp"
namespace olesnitskiy_v_dijkstra_crs {
class OlesnitskiyVDijkstraCrsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OlesnitskiyVDijkstraCrsSEQ(const InType &in);
 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<int> dijkstraCRS(const GraphCRS& graph);
  int findMinDistance(const std::vector<int>& distances, const std::vector<bool>& visited);
};
}  // namespace olesnitskiy_v_dijkstra_crs