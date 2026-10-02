// Small helpers shared by algorithm translation units.
#pragma once
#include "../core/bench_core.hpp"
namespace fb {
template <class T> inline constexpr bool is_record_v = std::is_same_v<T, KV> || std::is_same_v<T, Rec>;
template <class T> inline constexpr bool is_int_v = std::is_integral_v<T>;
template <class T> inline constexpr bool is_num_v = std::is_arithmetic_v<T>;
enum : int { kU = 1, kS = 2, kBoth = 3 };
}
