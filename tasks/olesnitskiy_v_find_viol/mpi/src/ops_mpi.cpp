#include "olesnitskiy_v_find_viol/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "olesnitskiy_v_find_viol/common/include/common.hpp"
#include "util/include/util.hpp"

namespace olesnitskiy_v_find_viol {

OlesnitskiyVFindViolMPI::OlesnitskiyVFindViolMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OlesnitskiyVFindViolMPI::ValidationImpl() {
  return true;
}

bool OlesnitskiyVFindViolMPI::PreProcessingImpl() {
  return true;
}

bool OlesnitskiyVFindViolMPI::RunImpl() {
  if (GetInput().size() < 2) {
    GetOutput() = 0;
    return true;
  }
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  const auto &input_data = GetInput();
  int total_size = static_cast<int>(input_data.size());

  if (total_size <= world_size) {
    int viol = 0;
    if (world_rank == 0) {
      const double epsilon = 1e-10;
      for (size_t i = 0; i < input_data.size() - 1; i++) {
        if (input_data[i] - input_data[i + 1] > epsilon) {
          viol++;
        }
      }
    }
    MPI_Bcast(&viol, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = viol;
    return true;
  }
  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displs(world_size, 0);
  std::vector<double> boundary_elements_for_scatter(world_size, 0.0);
  if (world_rank == 0) {
    int base_chunk = total_size / world_size;
    int remainder = total_size % world_size;
    int current_displ = 0;

    for (int i = 0; i < world_size; i++) {
      sendcounts[i] = base_chunk + (i < remainder ? 1 : 0);
      displs[i] = current_displ;

      if (i == 0) {
        boundary_elements_for_scatter[i] = 0.0;
      } else {
        int prev_process_last_idx = displs[i - 1] + sendcounts[i - 1] - 1;
        boundary_elements_for_scatter[i] = input_data[prev_process_last_idx];
      }

      current_displ += sendcounts[i];
    }
  }
  int my_chunk_size = 0;
  MPI_Scatter(sendcounts.data(), 1, MPI_INT, &my_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> local_data(my_chunk_size);
  MPI_Scatterv(input_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), my_chunk_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double prev_element;
  MPI_Scatter(boundary_elements_for_scatter.data(), 1, MPI_DOUBLE, &prev_element, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int local_viol = 0;
  const double epsilon = 1e-10;
  if (world_rank > 0 && my_chunk_size > 0) {
    if (prev_element - local_data[0] > epsilon) {
      local_viol++;
    }
  }
  if (my_chunk_size > 1) {
    for (int i = 0; i < my_chunk_size - 1; i++) {
      if (local_data[i] - local_data[i + 1] > epsilon) {
        local_viol++;
      }
    }
  }
  int total_viol;
  MPI_Allreduce(&local_viol, &total_viol, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_viol;

  return true;
}

bool OlesnitskiyVFindViolMPI::PostProcessingImpl() {
  return true;
}

}  // namespace olesnitskiy_v_find_viol
