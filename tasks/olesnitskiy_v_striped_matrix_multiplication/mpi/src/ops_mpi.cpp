#include "olesnitskiy_v_striped_matrix_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "util/include/util.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {
OlesnitskiyVStripedMatrixMultiplicationMPI::OlesnitskiyVStripedMatrixMultiplicationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0, 0, std::vector<double>());
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  rows_A_ = 0;
  cols_A_ = 0;
  rows_B_ = 0;
  cols_B_ = 0;
  rows_C_ = 0;
  cols_C_ = 0;
}

std::vector<int> OlesnitskiyVStripedMatrixMultiplicationMPI::calculate_counts(int total, int num_parts) {
  std::vector<int> counts(num_parts, 0);
  int base = total / num_parts;
  int remainder = total % num_parts;
  
  for (int i = 0; i < num_parts; ++i) {
    counts[i] = base + (i < remainder ? 1 : 0);
  }
  
  return counts;
}

std::vector<int> OlesnitskiyVStripedMatrixMultiplicationMPI::calculate_displacements(const std::vector<int>& counts) {
  std::vector<int> displs(counts.size(), 0);
  for (size_t i = 1; i < counts.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  return displs;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::ValidationImpl() {
  const auto& [rows_A, cols_A, data_A, rows_B, cols_B, data_B] = GetInput();
  rows_A_ = rows_A;
  cols_A_ = cols_A;
  data_A_ = data_A;
  rows_B_ = rows_B;
  cols_B_ = cols_B;
  data_B_ = data_B;
  if (rows_A == 0 || cols_A == 0 || rows_B == 0 || cols_B == 0) {
    return false;
  }
  
  if (data_A.size() != rows_A * cols_A || data_B.size() != rows_B * cols_B) {
    return false;
  }
  
  if (cols_A != rows_B) {
    return false;
  }
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::PreProcessingImpl() {
  rows_C_ = rows_A_;
  cols_C_ = cols_B_;
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::RunImpl() {
  if (static_cast<int>(rows_A_) < world_size_) {
    return RunOnSingleProcess();
  }
  auto row_counts = calculate_counts(static_cast<int>(rows_A_), world_size_);
  auto row_displs = calculate_displacements(row_counts);
  int rows_A_local = row_counts[rank_];
  std::vector<int> sendcounts_A(world_size_);
  std::vector<int> displs_A(world_size_);
  for (int i = 0; i < world_size_; ++i) {
    sendcounts_A[i] = row_counts[i] * static_cast<int>(cols_A_);
    displs_A[i] = row_displs[i] * static_cast<int>(cols_A_);
  }
  std::vector<double> local_A(rows_A_local * cols_A_);
  MPI_Scatterv(data_A_.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE, local_A.data(), sendcounts_A[rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_B(rows_B_ * cols_B_);
  if (rank_ == 0) {
    local_B = data_B_;
  }
  MPI_Bcast(local_B.data(), static_cast<int>(local_B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  std::vector<double> local_C(rows_A_local * cols_C_, 0.0);
  for (int local_row = 0; local_row < rows_A_local; ++local_row) {
    for (size_t col = 0; col < cols_C_; ++col) {
      double sum = 0.0;
      for (size_t k = 0; k < cols_A_; ++k) {
        sum += local_A[local_row * cols_A_ + k] * 
               local_B[k * cols_B_ + col];
      }
      local_C[local_row * cols_C_ + col] = sum;
    }
  }
  std::vector<int> recvcounts_C(world_size_);
  std::vector<int> displs_C(world_size_);
  for (int i = 0; i < world_size_; ++i) {
    recvcounts_C[i] = row_counts[i] * static_cast<int>(cols_C_);
    displs_C[i] = row_displs[i] * static_cast<int>(cols_C_);
  }
  if (rank_ == 0 && result_C_.empty()) {
    result_C_.resize(rows_C_ * cols_C_, 0.0);
  }
  MPI_Gatherv(local_C.data(), recvcounts_C[rank_], MPI_DOUBLE, rank_ == 0 ? result_C_.data() : nullptr,  recvcounts_C.data(), displs_C.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank_ == 0) {
    GetOutput() = std::make_tuple(rows_C_, cols_C_, result_C_);
    int result_rows = static_cast<int>(rows_C_);
    int result_cols = static_cast<int>(cols_C_);
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(result_C_.data(), static_cast<int>(result_C_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int result_rows = 0;
    int result_cols = 0;
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> received_result(result_rows * result_cols);
    MPI_Bcast(received_result.data(), static_cast<int>(received_result.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(static_cast<size_t>(result_rows), static_cast<size_t>(result_cols), received_result);
  }
  const auto& [out_rows, out_cols, out_data] = GetOutput();
  bool success = (out_rows == rows_C_) && (out_cols == cols_C_) && (out_data.size() == rows_C_ * cols_C_);
  MPI_Barrier(MPI_COMM_WORLD);
  return success;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::RunOnSingleProcess() {
  if (rank_ == 0) {
    result_C_.resize(rows_C_ * cols_C_, 0.0);
    for (size_t i = 0; i < rows_A_; ++i) {
      for (size_t j = 0; j < cols_B_; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < cols_A_; ++k) {
          sum += data_A_[i * cols_A_ + k] * data_B_[k * cols_B_ + j];
        }
        result_C_[i * cols_C_ + j] = sum;
      }
    }
    GetOutput() = std::make_tuple(rows_C_, cols_C_, result_C_);
    int result_rows = static_cast<int>(rows_C_);
    int result_cols = static_cast<int>(cols_C_);
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(result_C_.data(), static_cast<int>(result_C_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int result_rows = 0;
    int result_cols = 0;
    MPI_Bcast(&result_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> received_result(result_rows * result_cols);
    MPI_Bcast(received_result.data(), static_cast<int>(received_result.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(static_cast<size_t>(result_rows), static_cast<size_t>(result_cols), received_result);
  }
  const auto& [out_rows, out_cols, out_data] = GetOutput();
  bool success = (out_rows == rows_C_) && (out_cols == cols_C_) && (out_data.size() == rows_C_ * cols_C_);
  MPI_Barrier(MPI_COMM_WORLD);
  return success;
}

bool OlesnitskiyVStripedMatrixMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace olesnitskiy_v_striped_matrix_multiplication