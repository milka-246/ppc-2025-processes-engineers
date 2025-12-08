#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <iostream>
#include "olesnitskiy_v_dijkstra_crs/common/include/common.hpp"
#include "olesnitskiy_v_dijkstra_crs/mpi/include/ops_mpi.hpp"
#include "olesnitskiy_v_dijkstra_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace olesnitskiy_v_dijkstra_crs {

class OlesnitskiyVDijkstraCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_type = std::get<0>(params);
    std::string test_name = std::get<1>(params);
    switch (test_type) {
      case 0:
        createSingleVertexGraph();
        break;
      case 1:
        createTwoVerticesGraph();
        break;
      case 2:
        createChainGraph(5);
        break;
      case 3:
        createStarGraph(6);
        break;
      case 4:
        createCompleteGraph(4);
        break;
      case 5:
        createDisconnectedGraph();
        break;
      case 6:
        createGraphWithWeights();
        break;
      case 7:
        createRandomSparseGraph(10, 15);
        break;
      case 8:
        createRandomDenseGraph(8);
        break;
      case 9:
        createChainGraph(20);
        break;
      case 10:
        createMultiplePathsGraph();
        break;
      case 11:
        createZeroWeightGraph();
        break;
      case 12:
        createBinaryTreeGraph(3);
        break;
      case 13:
        createGridGraph(3, 3);
        break;
      default:
        createChainGraph(5);
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }
    if (static_cast<int>(output_data.size()) != expected_vertices_) {
      std::cout << "ERROR: Wrong size: expected " << expected_vertices_ 
                << ", got " << output_data.size() << std::endl;
      return false;
    }
    if (output_data[expected_source_] != 0) {
      std::cout << "ERROR: Source distance not 0: " << output_data[expected_source_] << std::endl;
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      int dist = output_data[i];
      if (dist < 0) {
        std::cout << "ERROR: Negative distance at vertex " << i << ": " << dist << std::endl;
        return false;
      }
    }
    if (!expected_distances_.empty()) {
      bool all_match = true;
      for (size_t i = 0; i < expected_distances_.size(); ++i) {
        int expected = expected_distances_[i];
        int actual = output_data[i];
        if (expected == std::numeric_limits<int>::max()) {
          if (actual != std::numeric_limits<int>::max()) {
            std::cout << "ERROR at vertex " << i << ": expected INF, got " << actual << std::endl;
            all_match = false;
          }
        } else if (actual != expected) {
          std::cout << "ERROR at vertex " << i << ": expected " << expected << ", got " << actual << std::endl;
          all_match = false;
        }
      }
      if (!all_match) {
        std::cout << "Expected distances: ";
        for (int d : expected_distances_) {
          if (d == std::numeric_limits<int>::max()) std::cout << "INF ";
          else std::cout << d << " ";
        }
        std::cout << "\nActual distances: ";
        for (int d : output_data) {
          if (d == std::numeric_limits<int>::max()) std::cout << "INF ";
          else std::cout << d << " ";
        }
        std::cout << std::endl;
        return false;
      }
    }
    
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_vertices_ = 0;
  int expected_source_ = 0;
  std::vector<int> expected_distances_;

  void createSingleVertexGraph() {
    std::vector<int> offsets = {0, 0};
    std::vector<int> edges;
    std::vector<int> weights;
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 1;
    expected_source_ = 0;
    expected_distances_ = {0};
  }

  void createTwoVerticesGraph() {
    std::vector<int> offsets = {0, 1, 2};
    std::vector<int> edges = {1, 0};
    std::vector<int> weights = {5, 5};
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 2;
    expected_source_ = 0;
    expected_distances_ = {0, 5};
  }

  void createChainGraph(int n) {
    std::vector<int> offsets(n + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    for (int i = 0; i < n; ++i) {
      offsets[i + 1] = offsets[i];
      if (i + 1 < n) {
        edges.push_back(i + 1);
        weights.push_back(1);
        offsets[i + 1]++;
      }
      if (i > 0) {
        edges.push_back(i - 1);
        weights.push_back(1);
        offsets[i + 1]++;
      }
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = n;
    expected_source_ = 0;
    expected_distances_.resize(n);
    for (int i = 0; i < n; ++i) {
      expected_distances_[i] = i;
    }
  }

  void createStarGraph(int n) {
    std::vector<int> offsets(n + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    int center = 0;
    for (int i = 1; i < n; ++i) {
      edges.push_back(i);
      weights.push_back(2);
    }
    offsets[1] = n - 1;
    for (int i = 1; i < n; ++i) {
      edges.push_back(center);
      weights.push_back(2);
      offsets[i + 1] = offsets[i] + 1;
    }
    input_data_ = std::make_tuple(center, offsets, edges, weights);
    expected_vertices_ = n;
    expected_source_ = center;
    expected_distances_.resize(n);
    expected_distances_[0] = 0;
    for (int i = 1; i < n; ++i) {
      expected_distances_[i] = 2;
    }
  }

  void createCompleteGraph(int n) {
    std::vector<int> offsets(n + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          edges.push_back(j);
          weights.push_back(1);
        }
      }
      offsets[i + 1] = offsets[i] + (n - 1);
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = n;
    expected_source_ = 0;
    expected_distances_.resize(n);
    expected_distances_[0] = 0;
    for (int i = 1; i < n; ++i) {
      expected_distances_[i] = 1;
    }
  }

  void createDisconnectedGraph() {
    std::vector<int> offsets = {0, 2, 4, 5, 5, 6};
    std::vector<int> edges = {1, 2, 0, 2, 0, 1, 4, 3};
    std::vector<int> weights(edges.size(), 1);
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 5;
    expected_source_ = 0;
    expected_distances_ = {0, 1, 1, std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
  }
  
  void createGraphWithWeights() {
    std::vector<int> offsets = {0, 2, 3, 4, 4};
    std::vector<int> edges = {1, 2, 2, 3, 2};
    std::vector<int> weights = {5, 2, 1, 3, 4};
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 4;
    expected_source_ = 0;
    expected_distances_ = {0, 5, 2, 5};
  }
  
  void createRandomSparseGraph(int vertices, int edges_count) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> vertex_dist(0, vertices - 1);
    std::uniform_int_distribution<> weight_dist(1, 10);
    std::vector<int> offsets(vertices + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    for (int i = 0; i < edges_count; ++i) {
      int u = vertex_dist(gen);
      int v = vertex_dist(gen);
      if (u != v) {
        edges.push_back(v);
        weights.push_back(weight_dist(gen));
        offsets[u + 1]++;
      }
    }
    for (int i = 0; i < vertices; ++i) {
      offsets[i + 1] += offsets[i];
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = vertices;
    expected_source_ = 0;
    expected_distances_.clear();
  }
  
  void createRandomDenseGraph(int vertices) {
    std::mt19937 gen(123);
    std::uniform_int_distribution<> weight_dist(1, 5);
    std::vector<int> offsets(vertices + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    for (int i = 0; i < vertices; ++i) {
      for (int j = 0; j < vertices; ++j) {
        if (i != j && (gen() % 10) < 7) {
          edges.push_back(j);
          weights.push_back(weight_dist(gen));
          offsets[i + 1]++;
        }
      }
    }
    for (int i = 0; i < vertices; ++i) {
      offsets[i + 1] += offsets[i];
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = vertices;
    expected_source_ = 0;
    expected_distances_.clear();
  }
  
  void createMultiplePathsGraph() {
    std::vector<int> offsets = {0, 2, 3, 4, 4};
    std::vector<int> edges = {1, 2, 3, 3, 1};
    std::vector<int> weights = {1, 1, 1, 1, 1};
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 4;
    expected_source_ = 0;
    expected_distances_ = {0, 1, 1, 2};
  }
  
  void createZeroWeightGraph() {
    std::vector<int> offsets = {0, 2, 4, 5, 5};
    std::vector<int> edges = {1, 2, 2, 3, 3, 2};
    std::vector<int> weights = {0, 1, 0, 2, 1, 0};
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = 4;
    expected_source_ = 0;
    expected_distances_ = {0, 0, 0, 1};
  }
  
  void createBinaryTreeGraph(int levels) {
    int vertices = (1 << levels) - 1;
    std::vector<int> offsets(vertices + 1, 0);
    std::vector<int> edges;
    std::vector<int> weights;
    std::vector<int> edge_counts(vertices, 0);
    for (int i = 0; i < vertices; ++i) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      if (left < vertices) {
        edge_counts[i]++;
      }
      if (right < vertices) {
        edge_counts[i]++;
      }
      if (i > 0) {
        edge_counts[i]++;
      }
    }
    offsets[0] = 0;
    for (int i = 0; i < vertices; ++i) {
      offsets[i + 1] = offsets[i] + edge_counts[i];
    }
    for (int i = 0; i < vertices; ++i) {
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      if (left < vertices) {
        edges.push_back(left);
        weights.push_back(1);
      }
      if (right < vertices) {
        edges.push_back(right);
        weights.push_back(1);
      }
      if (i > 0) {
        int parent = (i - 1) / 2;
        edges.push_back(parent);
        weights.push_back(1);
      }
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = vertices;
    expected_source_ = 0;
    expected_distances_.resize(vertices);
    for (int i = 0; i < vertices; ++i) {
      int depth = 0;
      int node = i;
      while (node > 0) {
        node = (node - 1) / 2;
        depth++;
      }
      expected_distances_[i] = depth;
    }
  }
  
  void createGridGraph(int rows, int cols) {
    int vertices = rows * cols;
    std::vector<int> offsets(vertices + 1, 0);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        int v = r * cols + c;
        if (c + 1 < cols) offsets[v + 1]++;
        if (r + 1 < rows) offsets[v + 1]++;
        if (c > 0) offsets[v + 1]++;
        if (r > 0) offsets[v + 1]++;
      }
    }
    for (int i = 0; i < vertices; ++i) {
      offsets[i + 1] += offsets[i];
    }
    std::vector<int> edges(offsets[vertices]);
    std::vector<int> weights(offsets[vertices]);
    std::vector<int> current_pos = offsets;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        int v = r * cols + c;
        if (c + 1 < cols) {
          int neighbor = v + 1;
          edges[current_pos[v]] = neighbor;
          weights[current_pos[v]] = 1;
          current_pos[v]++;
        }
        if (r + 1 < rows) {
          int neighbor = v + cols;
          edges[current_pos[v]] = neighbor;
          weights[current_pos[v]] = 1;
          current_pos[v]++;
        }
        if (c > 0) {
          int neighbor = v - 1;
          edges[current_pos[v]] = neighbor;
          weights[current_pos[v]] = 1;
          current_pos[v]++;
        }
        if (r > 0) {
          int neighbor = v - cols;
          edges[current_pos[v]] = neighbor;
          weights[current_pos[v]] = 1;
          current_pos[v]++;
        }
      }
    }
    input_data_ = std::make_tuple(0, offsets, edges, weights);
    expected_vertices_ = vertices;
    expected_source_ = 0;
    expected_distances_.resize(vertices);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        int v = r * cols + c;
        expected_distances_[v] = r + c;
      }
    }
  }
};

namespace {
TEST_P(OlesnitskiyVDijkstraCrsFuncTests, DijkstraCRSTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(0, "single_vertex"),
    std::make_tuple(1, "two_vertices"),
    std::make_tuple(2, "chain_5"),
    std::make_tuple(3, "star_6"),
    std::make_tuple(4, "complete_4"),
    std::make_tuple(5, "disconnected"),
    std::make_tuple(6, "weighted"),
    std::make_tuple(7, "random_sparse_10_15"),
    std::make_tuple(8, "random_dense_8"),
    std::make_tuple(9, "chain_20"),
    std::make_tuple(10, "multiple_paths"),
    std::make_tuple(11, "zero_weight"),
    std::make_tuple(12, "binary_tree_3"),
    std::make_tuple(13, "grid_3x3")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<OlesnitskiyVDijkstraCrsMPI, InType>(kTestParam, PPC_SETTINGS_olesnitskiy_v_dijkstra_crs),
                   ppc::util::AddFuncTask<OlesnitskiyVDijkstraCrsSEQ, InType>(kTestParam, PPC_SETTINGS_olesnitskiy_v_dijkstra_crs));
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = OlesnitskiyVDijkstraCrsFuncTests::PrintFuncTestName<OlesnitskiyVDijkstraCrsFuncTests>;
INSTANTIATE_TEST_SUITE_P(DijkstraCRSTests, OlesnitskiyVDijkstraCrsFuncTests, kGtestValues, kPerfTestName);
}  // namespace
}  // namespace olesnitskiy_v_dijkstra_crs