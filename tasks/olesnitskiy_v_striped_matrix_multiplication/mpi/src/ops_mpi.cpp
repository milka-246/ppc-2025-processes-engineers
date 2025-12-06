#include "olesnitskiy_v_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {

OlesnitskiyVStripedMatrixMultiplicationMPI::OlesnitskiyVStripedMatrixMultiplicationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0, 0, std::vector<double>());
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

std::vector<int> OlesnitskiyVStripedMatrixMultiplicationMPI::CalculateCounts(int total, int num_parts) {
  std::vector<int> counts(num_parts, 0);
  int base = total / num_parts;
  int remainder = total % num_parts;

  for (int i = 0; i < num_parts; ++i) {
    counts[i] = base + (i < remainder ? 1 : 0);
  }

  return counts;
}

std::vector<int> OlesnitskiyVStripedMatrixMultiplicationMPI::CalculateDisplacements(const std::vector<int> &counts) {
  std::vector<int> displs(counts.size(), 0);
  for (size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  return displs;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::ValidationImpl() {
  const auto &[rows_a, cols_a, data_a, rows_b, cols_b, data_b] = GetInput();
  rows_a_ = rows_a;
  cols_a_ = cols_a;
  data_a_ = data_a;
  rows_b_ = rows_b;
  cols_b_ = cols_b;
  data_b_ = data_b;
  if (rows_a == 0 || cols_a == 0 || rows_b == 0 || cols_b == 0) {
    return false;
  }

  if (data_a.size() != rows_a * cols_a || data_b.size() != rows_b * cols_b) {
    return false;
  }

  if (cols_a != rows_b) {
    return false;
  }
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::PreProcessingImpl() {
  rows_c_ = rows_a_;
  cols_c_ = cols_b_;
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::RunImpl() {
  if (std::cmp_less(rows_a_, world_size_)) {
    return RunOnSingleProcess();
  }
  auto row_counts = CalculateCounts(static_cast<int>(rows_a_), world_size_);
  auto row_displs = CalculateDisplacements(row_counts);
  int rows_a_local = row_counts[rank_];
  std::vector<int> sendcounts_a(world_size_);
  std::vector<int> displs_a(world_size_);
  for (int i = 0; i < world_size_; ++i) {
    sendcounts_a[i] = row_counts[i] * static_cast<int>(cols_a_);
    displs_a[i] = row_displs[i] * static_cast<int>(cols_a_);
  }
  std::vector<double> local_a(rows_a_local * cols_a_);
  MPI_Scatterv(data_a_.data(), sendcounts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), sendcounts_a[rank_],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_b(rows_b_ * cols_b_);
  if (rank_ == 0) {
    local_b = data_b_;
  }
  MPI_Bcast(local_b.data(), static_cast<int>(local_b.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_c(rows_a_local * cols_c_, 0.0);
  for (int local_row = 0; local_row < rows_a_local; ++local_row) {
    for (size_t col = 0; col < cols_c_; ++col) {
      double sum = 0.0;
      for (size_t k = 0; k < cols_a_; ++k) {
        sum += local_a[(local_row * cols_a_) + k] * local_b[(k * cols_b_) + col];
      }
      local_c[(local_row * cols_c_) + col] = sum;
    }
  }
  std::vector<int> recvcounts_c(world_size_);
  std::vector<int> displs_c(world_size_);
  for (int i = 0; i < world_size_; ++i) {
    recvcounts_c[i] = row_counts[i] * static_cast<int>(cols_c_);
    displs_c[i] = row_displs[i] * static_cast<int>(cols_c_);
  }

  const std::size_t result_size = rows_c_ * cols_c_;
  if (rank_ == 0 && result_size > 0) {
    result_c_.resize(result_size, 0.0);
  }

  MPI_Gatherv(local_c.data(), recvcounts_c[rank_], MPI_DOUBLE,
              rank_ == 0 && result_size > 0 ? result_c_.data() : nullptr, recvcounts_c.data(), displs_c.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    if (result_c_.empty()) {
      GetOutput() = std::make_tuple(0UL, 0UL, std::vector<double>());
    } else {
      GetOutput() = std::make_tuple(rows_c_, cols_c_, result_c_);
    }
    int result_rows = static_cast<int>(rows_c_);
    int result_cols = static_cast<int>(cols_c_);
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!result_c_.empty()) {
      MPI_Bcast(result_c_.data(), static_cast<int>(result_c_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    int result_rows = 0;
    int result_cols = 0;
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_rows > 0 && result_cols > 0) {
      std::vector<double> received_result(static_cast<std::size_t>(result_rows) *
                                          static_cast<std::size_t>(result_cols));
      MPI_Bcast(received_result.data(), static_cast<int>(received_result.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GetOutput() =
          std::make_tuple(static_cast<size_t>(result_rows), static_cast<size_t>(result_cols), received_result);
    } else {
      GetOutput() = std::make_tuple(0UL, 0UL, std::vector<double>());
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::RunOnSingleProcess() {
  const std::size_t result_size = rows_c_ * cols_c_;

  if (rank_ == 0) {
    if (result_size > 0) {
      result_c_.resize(result_size, 0.0);
      for (size_t i = 0; i < rows_a_; ++i) {
        for (size_t j = 0; j < cols_b_; ++j) {
          double sum = 0.0;
          for (size_t k = 0; k < cols_a_; ++k) {
            sum += data_a_[(i * cols_a_) + k] * data_b_[(k * cols_b_) + j];
          }
          result_c_[(i * cols_c_) + j] = sum;
        }
      }
    }

    if (result_c_.empty()) {
      GetOutput() = std::make_tuple(0UL, 0UL, std::vector<double>());
    } else {
      GetOutput() = std::make_tuple(rows_c_, cols_c_, result_c_);
    }

    int result_rows = static_cast<int>(rows_c_);
    int result_cols = static_cast<int>(cols_c_);
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!result_c_.empty()) {
      MPI_Bcast(result_c_.data(), static_cast<int>(result_c_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    int result_rows = 0;
    int result_cols = 0;
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (result_rows > 0 && result_cols > 0) {
      std::vector<double> received_result(static_cast<std::size_t>(result_rows) *
                                          static_cast<std::size_t>(result_cols));
      MPI_Bcast(received_result.data(), static_cast<int>(received_result.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      GetOutput() =
          std::make_tuple(static_cast<size_t>(result_rows), static_cast<size_t>(result_cols), received_result);
    } else {
      GetOutput() = std::make_tuple(0UL, 0UL, std::vector<double>());
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace olesnitskiy_v_striped_matrix_multiplication
