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

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int idx = 0; idx < size; ++idx) {
    counts[idx] = (vertices / size) + (idx < (vertices % size) ? 1 : 0);
    displs[idx] = (idx == 0) ? 0 : displs[idx - 1] + counts[idx - 1];
  }

  int start_idx = displs[rank];
  int end_idx = start_idx + counts[rank];
  int local_vertices = counts[rank];

  std::vector<int> local_distances(local_vertices, std::numeric_limits<int>::max());
  std::vector<bool> local_visited(local_vertices, false);

  bool source_is_local = (source >= start_idx && source < end_idx);
  if (source_is_local) {
    local_distances[source - start_idx] = 0;
  }

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
  if (source_is_local) {
    pq.emplace(0, source);
  }

  struct Update {
    int vertex;
    int distance;
  };

  std::vector<std::vector<Update>> send_bufs(size);
  int active = 1;

  while (active > 0) {
    int local_best_dist = std::numeric_limits<int>::max();
    int local_best_vertex = -1;

    if (!pq.empty()) {
      while (!pq.empty() && local_visited[pq.top().second - start_idx]) {
        pq.pop();
      }
      if (!pq.empty()) {
        local_best_dist = pq.top().first;
        local_best_vertex = pq.top().second;
      }
    }

    struct DistVertexPair {
      int dist;
      int vertex;
    } local_info = {.dist = local_best_dist, .vertex = local_best_vertex}, global_info = {};

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
      if (!pq.empty() && pq.top().second == global_info.vertex) {
        pq.pop();
      }

      int vertex = global_info.vertex;
      int start = offsets[vertex];
      int end = offsets[vertex + 1];

      for (int i = start; i < end; ++i) {
        int neighbor = edges[i];
        int weight = weights[i];
        int new_dist = global_info.dist + weight;

        int owner = 0;
        for (int j = 0; j < size; ++j) {
          if (neighbor >= displs[j] && neighbor < (displs[j] + counts[j])) {
            owner = j;
            break;
          }
        }

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

    for (int i = 0; i < size; ++i) {
      send_displs[i] = total_send;
      recv_displs[i] = total_recv;
      total_send += send_sizes[i];
      total_recv += recv_sizes[i];
    }

    const auto send_data_size = static_cast<std::size_t>(total_send) * 2;
    const auto recv_data_size = static_cast<std::size_t>(total_recv) * 2;
    std::vector<int> send_data(send_data_size);
    std::vector<int> recv_data(recv_data_size);

    int idx = 0;
    for (int i = 0; i < size; ++i) {
      for (const auto &update : send_bufs[i]) {
        send_data[idx++] = update.vertex;
        send_data[idx++] = update.distance;
      }
      send_bufs[i].clear();
    }

    std::vector<int> send_counts_bytes(size);
    std::vector<int> recv_counts_bytes(size);
    std::vector<int> send_displs_bytes(size);
    std::vector<int> recv_displs_bytes(size);

    for (int i = 0; i < size; ++i) {
      send_counts_bytes[i] = send_sizes[i] * 2;
      recv_counts_bytes[i] = recv_sizes[i] * 2;
      send_displs_bytes[i] = send_displs[i] * 2;
      recv_displs_bytes[i] = recv_displs[i] * 2;
    }

    MPI_Alltoallv(send_data.data(), send_counts_bytes.data(), send_displs_bytes.data(), MPI_INT, recv_data.data(),
                  recv_counts_bytes.data(), recv_displs_bytes.data(), MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv * 2; i += 2) {
      int neighbor = recv_data[i];
      int new_dist = recv_data[i + 1];

      if (neighbor >= start_idx && neighbor < end_idx) {
        int local_idx = neighbor - start_idx;
        if (!local_visited[local_idx] && new_dist < local_distances[local_idx]) {
          local_distances[local_idx] = new_dist;
          pq.emplace(new_dist, neighbor);
        }
      }
    }

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
