/* scandum crumsort / fluxsort / quadsort wrapper, compiled as C.
   FB_SCANDUM selects the header: 0 crumsort, 1 fluxsort, 2 quadsort. */
#include <stddef.h>
#include <stdint.h>
#ifndef FB_SCANDUM
#  define FB_SCANDUM 0
#endif
#if FB_SCANDUM == 0
#  include "crumsort.h"
#  define FB_GEN crumsort
#  define FB_PRIM crumsort_prim
#elif FB_SCANDUM == 1
#  include "fluxsort.h"
#  define FB_GEN fluxsort
#  define FB_PRIM fluxsort_prim
#else
#  include "quadsort.h"
#  define FB_GEN quadsort
#  define FB_PRIM quadsort_prim
#endif
static int cmp_f32(const void* a, const void* b) { return *(const float*)a > *(const float*)b; }
static int cmp_f64(const void* a, const void* b) { return *(const double*)a > *(const double*)b; }
typedef struct { uint32_t key, id; } fb_rec;
static int cmp_rec(const void* a, const void* b) { return ((const fb_rec*)a)->key > ((const fb_rec*)b)->key; }
void fb_scandum_i32(void* p, size_t n) { FB_PRIM(p, n, 4); }
void fb_scandum_u32(void* p, size_t n) { FB_PRIM(p, n, 5); }
void fb_scandum_i64(void* p, size_t n) { FB_PRIM(p, n, 8); }
void fb_scandum_u64(void* p, size_t n) { FB_PRIM(p, n, 9); }
void fb_scandum_f32(void* p, size_t n) { FB_GEN(p, n, sizeof(float), cmp_f32); }
void fb_scandum_f64(void* p, size_t n) { FB_GEN(p, n, sizeof(double), cmp_f64); }
void fb_scandum_rec(void* p, size_t n) { FB_GEN(p, n, sizeof(fb_rec), cmp_rec); }
