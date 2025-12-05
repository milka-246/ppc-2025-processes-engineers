#include "olesnitskiy_v_striped_matrix_multiplication/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "olesnitskiy_v_striped_matrix_multiplication/common/include/common.hpp"
#include "util/include/util.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {
OlesnitskiyVStripedMatrixMultiplicationSEQ::OlesnitskiyVStripedMatrixMultiplicationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0, 0, std::vector<double>());
  
  rows_A_ = 0;
  cols_A_ = 0;
  rows_B_ = 0;
  cols_B_ = 0;
  rows_C_ = 0;
  cols_C_ = 0;
  num_stripes_ = 1;
}

int OlesnitskiyVStripedMatrixMultiplicationSEQ::find_common_divisor(int a, int b, int max_divisor) const {
  if (a <= 0 || b <= 0 || max_divisor <= 1) return 1;
  
  for (int d = std::min({a, b, max_divisor}); d >= 1; --d) {
    if (a % d == 0 && b % d == 0) {
      return d;
    }
  }
  
  return 1;
}

bool OlesnitskiyVStripedMatrixMultiplicationSEQ::ValidationImpl() {
  const auto& [rows_A, cols_A, data_A, rows_B, cols_B, data_B] = GetInput();
  const auto& [out_rows, out_cols, out_data] = GetOutput();
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
  if (out_rows != 0 || out_cols != 0 || !out_data.empty()) {
    return false;
  }
  
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationSEQ::PreProcessingImpl() {
  rows_C_ = rows_A_;
  cols_C_ = cols_B_;
  int max_stripes = 8;
  num_stripes_ = find_common_divisor(static_cast<int>(rows_A_), static_cast<int>(cols_B_), max_stripes);
  if (num_stripes_ < 2) {
    num_stripes_ = 1;
  }
  result_C_.resize(rows_C_ * cols_C_, 0.0);
  GetOutput() = std::make_tuple(0, 0, std::vector<double>());
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationSEQ::RunImpl() {
  if (rows_A_ == 0 || cols_B_ == 0) {
    return false;
  }
  if (num_stripes_ == 1) {
    for (size_t i = 0; i < rows_A_; ++i) {
      for (size_t j = 0; j < cols_B_; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < cols_A_; ++k) {
          sum += data_A_[i * cols_A_ + k] * data_B_[k * cols_B_ + j];
        }
        result_C_[i * cols_B_ + j] = sum;
      }
    }
  } else {
    if (rows_A_ % num_stripes_ != 0 || cols_B_ % num_stripes_ != 0) {
      return false;
    }
    size_t rows_per_stripe = rows_A_ / num_stripes_;
    size_t cols_per_stripe = cols_B_ / num_stripes_;
    for (int stripe_a = 0; stripe_a < num_stripes_; ++stripe_a) {
      size_t start_row_a = stripe_a * rows_per_stripe;
      for (int stripe_b = 0; stripe_b < num_stripes_; ++stripe_b) {
        size_t start_col_b = stripe_b * cols_per_stripe;
        for (size_t i = 0; i < rows_per_stripe; ++i) {
          size_t row_idx = start_row_a + i;
          for (size_t j = 0; j < cols_per_stripe; ++j) {
            size_t col_idx = start_col_b + j;
            double sum = 0.0;
            for (size_t k = 0; k < cols_A_; ++k) {
              sum += data_A_[row_idx * cols_A_ + k] * 
                     data_B_[k * cols_B_ + col_idx];
            }
            result_C_[row_idx * cols_B_ + col_idx] = sum;
          }
        }
      }
    }
  }
  GetOutput() = std::make_tuple(rows_C_, cols_C_, result_C_);
  return true;
}

bool OlesnitskiyVStripedMatrixMultiplicationSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace olesnitskiy_v_striped_matrix_multiplication