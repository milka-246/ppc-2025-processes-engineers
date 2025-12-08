#include "olesnitskiy_v_dijkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "olesnitskiy_v_dijkstra_crs/common/include/common.hpp"

namespace olesnitskiy_v_dijkstra_crs {

OlesnitskiyVDijkstraCrsMPI::OlesnitskiyVDijkstraCrsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool OlesnitskiyVDijkstraCrsMPI::ValidationImpl() {
  const auto &input = GetInput();
  int source = std::get<0>(input);
  const auto &offsets = std::get<1>(input);
  const auto &edges = std::get<2>(input);
  const auto &weights = std::get<3>(input);

  if (offsets.empty()) {
    return false;
  }
  int vertices = static_cast<int>(offsets.size()) - 1;
  if (vertices <= 0) {
    return false;
  }
  if (source < 0 || source >= vertices) {
    return false;
  }
  if (edges.size() != weights.size()) {
    return false;
  }

  return true;
}

bool OlesnitskiyVDijkstraCrsMPI::PreProcessingImpl() {
  return true;
}

namespace {
struct Update {
  int vertex;
  int distance;
};

struct DistVertexPair {
  int dist;
  int vertex;
};

int FindOwner(int vertex, const std::vector<int> &displs, const std::vector<int> &counts, int size) {
  for (int j = 0; j < size; ++j) {
    if (vertex >= displs[j] && vertex < (displs[j] + counts[j])) {
      return j;
    }
  }
  return 0;
}

void ProcessLocalVertex(int vertex, int distance, const std::vector<int> &offsets, const std::vector<int> &edges,
                        const std::vector<int> &weights, std::vector<int> &local_distances,
                        std::vector<bool> &local_visited,
                        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> &pq,
                        int start_idx, const std::vector<int> &displs, const std::vector<int> &counts, int rank,
                        int size, std::vector<std::vector<Update>> &send_bufs) {
  int start = offsets[vertex];
  int end = offsets[vertex + 1];

  for (int i = start; i < end; ++i) {
    int neighbor = edges[i];
    int weight = weights[i];
    int new_dist = distance + weight;

    int owner = FindOwner(neighbor, displs, counts, size);

    if (owner == rank) {
      int neighbor_local_idx = neighbor - start_idx;
      if (!local_visited[neighbor_local_idx] && new_dist < local_distances[neighbor_local_idx]) {
        local_distances[neighbor_local_idx] = new_dist;
        pq.emplace(new_dist, neighbor);
      }
    } else {
      send_bufs[owner].push_back(Update{.vertex = neighbor, .distance = new_dist});
    }
  }
}

void PrepareSendData(const std::vector<std::vector<Update>> &send_bufs, std::vector<int> &send_data) {
  int idx = 0;
  for (size_t i = 0; i < send_bufs.size(); ++i) {
    for (const auto &update : send_bufs[i]) {
      send_data[idx++] = update.vertex;
      send_data[idx++] = update.distance;
    }
  }
}

void ProcessReceivedData(
    const std::vector<int> &recv_data, int total_recv, int start_idx, int end_idx, std::vector<int> &local_distances,
    std::vector<bool> &local_visited,
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> &pq) {
  for (int i = 0; i < total_recv * 2; i += 2) {
    int neighbor = recv_data[i];
    int new_dist = recv_data[i + 1];

    bool is_local = (neighbor >= start_idx && neighbor < end_idx);
    if (!is_local) {
      continue;
    }

    int local_idx = neighbor - start_idx;
    bool should_update = !local_visited[local_idx] && new_dist < local_distances[local_idx];
    if (!should_update) {
      continue;
    }

    local_distances[local_idx] = new_dist;
    pq.emplace(new_dist, neighbor);
  }
}

void CalculateDisplacements(const std::vector<int> &sizes, std::vector<int> &displs, int &total) {
  total = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    displs[i] = total;
    total += sizes[i];
  }
}

void PrepareByteArrays(const std::vector<int> &sizes, const std::vector<int> &displs, std::vector<int> &counts_bytes,
                       std::vector<int> &displs_bytes) {
  for (size_t i = 0; i < sizes.size(); ++i) {
    counts_bytes[i] = sizes[i] * 2;
    displs_bytes[i] = displs[i] * 2;
  }
}

void ExchangeUpdates(std::vector<std::vector<Update>> &send_bufs, std::vector<int> &local_distances,
                     std::vector<bool> &local_visited,
                     std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> &pq,
                     int start_idx, int end_idx) {
  int size = static_cast<int>(send_bufs.size());
  std::vector<int> send_sizes(size);
  std::vector<int> recv_sizes(size);

  for (int i = 0; i < size; ++i) {
    send_sizes[i] = static_cast<int>(send_bufs[i].size());
  }

  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> send_displs(size);
  std::vector<int> recv_displs(size);
  int total_send = 0;
  int total_recv = 0;

  CalculateDisplacements(send_sizes, send_displs, total_send);
  CalculateDisplacements(recv_sizes, recv_displs, total_recv);

  const auto send_data_size = static_cast<std::size_t>(total_send) * 2;
  const auto recv_data_size = static_cast<std::size_t>(total_recv) * 2;
  std::vector<int> send_data(send_data_size);
  std::vector<int> recv_data(recv_data_size);

  PrepareSendData(send_bufs, send_data);

  for (int i = 0; i < size; ++i) {
    send_bufs[i].clear();
  }

  std::vector<int> send_counts_bytes(size);
  std::vector<int> recv_counts_bytes(size);
  std::vector<int> send_displs_bytes(size);
  std::vector<int> recv_displs_bytes(size);

  PrepareByteArrays(send_sizes, send_displs, send_counts_bytes, send_displs_bytes);
  PrepareByteArrays(recv_sizes, recv_displs, recv_counts_bytes, recv_displs_bytes);

  MPI_Alltoallv(send_data.data(), send_counts_bytes.data(), send_displs_bytes.data(), MPI_INT, recv_data.data(),
                recv_counts_bytes.data(), recv_displs_bytes.data(), MPI_INT, MPI_COMM_WORLD);

  ProcessReceivedData(recv_data, total_recv, start_idx, end_idx, local_distances, local_visited, pq);
}

void InitializeLocalData(
    int vertices, int size, int rank, std::vector<int> &counts, std::vector<int> &displs, int &start_idx, int &end_idx,
    int &local_vertices, int source, std::vector<int> &local_distances, std::vector<bool> &local_visited,
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> &pq) {
  counts.resize(size);
  displs.resize(size);
  for (int idx = 0; idx < size; ++idx) {
    counts[idx] = (vertices / size) + (idx < (vertices % size) ? 1 : 0);
    displs[idx] = (idx == 0) ? 0 : displs[idx - 1] + counts[idx - 1];
  }

  start_idx = displs[rank];
  end_idx = start_idx + counts[rank];
  local_vertices = counts[rank];

  local_distances.resize(local_vertices, std::numeric_limits<int>::max());
  local_visited.resize(local_vertices, false);

  bool source_is_local = (source >= start_idx && source < end_idx);
  if (source_is_local) {
    local_distances[source - start_idx] = 0;
    pq.emplace(0, source);
  }
}

}  // namespace

bool OlesnitskiyVDijkstraCrsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int vertices = 0;
  int source = 0;
  std::vector<int> offsets;
  std::vector<int> edges;
  std::vector<int> weights;

  if (rank == 0) {
    const auto &input = GetInput();
    source = std::get<0>(input);
    offsets = std::get<1>(input);
    edges = std::get<2>(input);
    weights = std::get<3>(input);
    vertices = static_cast<int>(offsets.size()) - 1;
  }

  MPI_Bcast(&vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    offsets.resize(vertices + 1);
  }
  MPI_Bcast(offsets.data(), vertices + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total_edges = 0;
  if (rank == 0) {
    total_edges = static_cast<int>(edges.size());
  }
  MPI_Bcast(&total_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    edges.resize(total_edges);
    weights.resize(total_edges);
  }
  MPI_Bcast(edges.data(), total_edges, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(weights.data(), total_edges, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts;
  std::vector<int> displs;
  int start_idx = 0;
  int end_idx = 0;
  int local_vertices = 0;
  std::vector<int> local_distances;
  std::vector<bool> local_visited;
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;

  InitializeLocalData(vertices, size, rank, counts, displs, start_idx, end_idx, local_vertices, source, local_distances,
                      local_visited, pq);

  std::vector<std::vector<Update>> send_bufs(size);
  int active = 1;

  while (active > 0) {
    int local_best_dist = std::numeric_limits<int>::max();
    int local_best_vertex = -1;

    if (!pq.empty()) {
      local_best_dist = pq.top().first;
      local_best_vertex = pq.top().second;
      int local_idx = local_best_vertex - start_idx;
      if (local_visited[local_idx]) {
        pq.pop();
        continue;
      }
    }

    DistVertexPair local_info = {.dist = local_best_dist, .vertex = local_best_vertex};
    DistVertexPair global_info = {};

    MPI_Allreduce(&local_info, &global_info, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (global_info.vertex == -1 || global_info.dist == std::numeric_limits<int>::max()) {
      break;
    }

    bool global_vertex_is_local = (global_info.vertex >= start_idx && global_info.vertex < end_idx);
    if (global_vertex_is_local) {
      int local_idx = global_info.vertex - start_idx;
      if (local_visited[local_idx]) {
        continue;
      }
      local_visited[local_idx] = true;
      pq.pop();

      ProcessLocalVertex(global_info.vertex, global_info.dist, offsets, edges, weights, local_distances, local_visited,
                         pq, start_idx, displs, counts, rank, size, send_bufs);
    }

    ExchangeUpdates(send_bufs, local_distances, local_visited, pq, start_idx, end_idx);

    int local_active = !pq.empty() ? 1 : 0;
    MPI_Allreduce(&local_active, &active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    std::vector<int> global_distances(vertices, std::numeric_limits<int>::max());
    std::ranges::copy(local_distances.begin(), local_distances.end(), global_distances.begin() + start_idx);

    for (int src = 1; src < size; ++src) {
      MPI_Recv(global_distances.data() + displs[src], counts[src], MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    GetOutput() = global_distances;
  } else {
    MPI_Send(local_distances.data(), local_vertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
    GetOutput() = std::vector<int>();
  }

  return true;
}

bool OlesnitskiyVDijkstraCrsMPI::PostProcessingImpl() {
  return true;
}
}  // namespace olesnitskiy_v_dijkstra_crs
