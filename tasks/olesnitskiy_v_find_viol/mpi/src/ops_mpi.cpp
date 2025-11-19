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
  std::vector<double> expanded_data;
  if (world_rank == 0) {
    int base_chunk = total_size / world_size;
    int remainder = total_size % world_size;
    int current_displ = 0;
    for (int i = 0; i < world_size; i++) {
      sendcounts[i] = base_chunk + (i < remainder ? 1 : 0);
      displs[i] = current_displ;
      current_displ += sendcounts[i];
    }
    expanded_data.resize(total_size + world_size - 1);
    int idx = 0;
    for (int proc = 0; proc < world_size; proc++) {
      int start = displs[proc];
      int end = start + sendcounts[proc];

      for (int i = start; i < end; i++) {
        expanded_data[idx++] = input_data[i];
      }

      if (proc < world_size - 1 && end < total_size) {
        expanded_data[idx++] = input_data[end - 1];
      }
      if (proc != 0) {
        sendcounts[proc]++;
      }
    }

    displs[0] = 0;
    for (int i = 1; i < world_size; i++) {
      displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
  }

  MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  int my_chunk_size = sendcounts[world_rank];
  if (my_chunk_size < 0) {
    // Обработка ошибки
    GetOutput() = 0;
    return true;
  }

  std::vector<double> local_data(my_chunk_size);

  // Распределение данных
  MPI_Scatterv(expanded_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), my_chunk_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Локальные вычисления
  int local_viol = 0;
  const double epsilon = 1e-10;

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
