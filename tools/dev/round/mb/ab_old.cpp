// ab_old: a second fyx header (e.g. HEAD's) under namespace fyx_old, for
// same-process A/B timing.  Build with -I<dir holding the old fyx_sort.hpp>.
#define fyx fyx_old
#define fyx_sort_int32 fyx_old_sort_int32
#define fyx_sort_double fyx_old_sort_double
#include "fyx_sort.hpp"
#include <cstdint>
template <class T> static void s(T* p, std::size_t n) { fyx_old::Options o; o.parallel = fyx_old::Tri::Off; fyx_old::sort(p, n, o); }
void old_sort(std::int32_t* p, std::size_t n) { s(p, n); }
void old_sort(std::uint32_t* p, std::size_t n) { s(p, n); }
void old_sort(std::int64_t* p, std::size_t n) { s(p, n); }
void old_sort(std::uint64_t* p, std::size_t n) { s(p, n); }
void old_sort(float* p, std::size_t n) { s(p, n); }
void old_sort(double* p, std::size_t n) { s(p, n); }
