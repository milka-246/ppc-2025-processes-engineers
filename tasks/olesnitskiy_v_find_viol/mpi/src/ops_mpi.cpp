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
  const auto &input_data = GetInput();
  int total_size = input_data.size();
  const double epsilon = 1e-10;
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
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
  int base_chunk = total_size / world_size;
  int remainder = total_size % world_size;
  int my_start = world_rank * base_chunk + std::min(world_rank, remainder);
  int my_end = my_start + base_chunk + (world_rank < remainder ? 1 : 0);
  int local_viol = 0;
  for (int i = my_start; i < my_end - 1; i++) {
    if (input_data[i] - input_data[i + 1] > epsilon) {
      local_viol++;
    }
  }
  if (world_rank > 0) {
    if (input_data[my_start - 1] - input_data[my_start] > epsilon) {
      local_viol++;
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
