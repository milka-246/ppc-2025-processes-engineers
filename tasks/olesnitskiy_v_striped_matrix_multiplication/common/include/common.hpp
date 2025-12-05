#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace olesnitskiy_v_striped_matrix_multiplication {

// Вход: две матрицы с их размерами
using InType = std::tuple<
    size_t, size_t, std::vector<double>,  // A: rows, cols, data
    size_t, size_t, std::vector<double>   // B: rows, cols, data
>;

// Выход: результирующая матрица C
using OutType = std::tuple<
    size_t, size_t, std::vector<double>   // C: rows, cols, data
>;

using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace olesnitskiy_v_striped_matrix_multiplication