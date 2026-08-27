#include "../fyx_sort.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <functional>
using namespace fyx::detail;

static int failures = 0;

template<class T>
void check(const char* tag, std::vector<T> a){
    std::vector<T> ref = a;
    std::sort(ref.begin(), ref.end());
    std::vector<T> got = a;
    if(!radix_sort(got.data(), got.size())){ printf("  %s: scratch fail\n", tag); failures++; return; }
    // For floats compare bitwise-insensitively via total order: NaNs sort to ends.
    bool ok = true;
    if(got.size()!=ref.size()) ok=false;
    for(size_t i=0;i<got.size() && ok;i++){
        if(got[i]!=ref[i] && !(got[i]!=got[i] && ref[i]!=ref[i])) ok=false;
    }
    if(!ok){ printf("  MISMATCH %s (n=%zu)\n", tag, a.size()); failures++; }
}

template<class T>
void suite(const char* name, std::mt19937_64& rng){
    printf("%s\n", name);
    size_t sizes[] = {0,1,2,3,7,8,15,16,17,63,64,65,100,255,256,257,1000,4096,10000,65536,100000};
    for(size_t n : sizes){
        std::vector<T> v(n);
        // random
        for(size_t i=0;i<n;i++){ uint64_t r=rng(); T x; std::memcpy(&x,&r,sizeof(T)<sizeof(r)?sizeof(T):sizeof(T)); v[i]=x; }
        if(std::is_floating_point<T>::value){ for(size_t i=0;i<n;i++) if(v[i]!=v[i]) v[i]=T(0); }
        check<T>("random", v);
        // sorted
        std::sort(v.begin(), v.end()); check<T>("sorted", v);
        // reverse
        std::reverse(v.begin(), v.end()); check<T>("reverse", v);
        // all equal
        std::vector<T> e(n, T(42)); check<T>("allequal", e);
        // many duplicates
        std::vector<T> d(n); for(size_t i=0;i<n;i++) d[i]=T(rng()%5); check<T>("dups", d);
        // extremes
        std::vector<T> x(n);
        for(size_t i=0;i<n;i++){
            switch(rng()%4){
                case 0: x[i]=std::numeric_limits<T>::max(); break;
                case 1: x[i]=std::numeric_limits<T>::lowest(); break;
                case 2: x[i]=T(0); break;
                default: x[i]=T(-1);
            }
        }
        check<T>("extremes", x);
    }
}

int main(){
    std::mt19937_64 rng(12345);
    suite<int32_t>("int32", rng);
    suite<uint32_t>("uint32", rng);
    suite<int64_t>("int64", rng);
    suite<uint64_t>("uint64", rng);
    suite<float>("float", rng);
    suite<double>("double", rng);

#if FYX_ENABLE_PARALLEL
    // High-entropy 64-bit default-order inputs use the high-prefix radix path
    // before the ordinary 8-pass radix fallback.  Exercise both ascending and
    // descending top-level dispatch on ranges large enough to trigger it.
    if (parallel_available()) {
        printf("parallel high-prefix 64-bit random\n");
        const size_t n = (size_t(1) << 20) + 123;
        {
            std::vector<uint64_t> v(n);
            for (auto& x : v) x = rng();
            fyx::sort(v.data(), v.size());
            bool ok = std::is_sorted(v.begin(), v.end());
            printf("  uint64 ascending: %s\n", ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }
        {
            std::vector<int64_t> v(n);
            for (auto& x : v) x = static_cast<int64_t>(rng());
            fyx::sort(v.data(), v.size(), std::greater<int64_t>{});
            bool ok = std::is_sorted(v.begin(), v.end(), std::greater<int64_t>{});
            printf("  int64 descending: %s\n", ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }
        {
            std::vector<double> v(n);
            for (auto& x : v) x = std::generate_canonical<double, 53>(rng);
            for (size_t i = 0; i < 256 && i * 1024 + 2 < v.size(); ++i) {
                v[i * 1024 + 0] = std::numeric_limits<double>::quiet_NaN();
                v[i * 1024 + 1] = -0.0;
                v[i * 1024 + 2] = +0.0;
            }
            fyx::sort(v.data(), v.size());
            bool ok = true;
            size_t nan0 = 0;
            while (nan0 < v.size() && v[nan0] == v[nan0]) ++nan0;
            for (size_t i = 1; i < nan0; ++i) if (v[i] < v[i - 1]) ok = false;
            for (size_t i = nan0; i < v.size(); ++i) if (v[i] == v[i]) ok = false;
            auto zr = std::equal_range(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(nan0), 0.0);
            bool seen_pos_zero = false;
            for (auto it = zr.first; it != zr.second; ++it) {
                if (std::signbit(*it)) {
                    if (seen_pos_zero) ok = false;
                } else {
                    seen_pos_zero = true;
                }
            }
            printf("  double ascending: %s\n", ok ? "OK" : "FAIL");
            if (!ok) failures++;
        }
    }
#endif

    // float special values including NaN, +-0, +-inf
    printf("float specials\n");
    {
        std::vector<float> v = {1.0f,-1.0f,0.0f,-0.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::denorm_min(),
            3.14159f, -2.71828f, 1e30f, -1e30f};
        std::vector<float> g=v;
        radix_sort(g.data(), g.size());
        // verify non-NaN portion is nondecreasing and NaN at the end
        bool ok=true; size_t i=0;
        for(; i<g.size() && g[i]==g[i]; ++i) if(i && g[i-1]>g[i]) ok=false;
        for(size_t j=i;j<g.size();++j) if(g[j]==g[j]) ok=false;
        printf("  ordering+NaN-at-end: %s\n", ok?"OK":"FAIL");
        if(!ok) failures++;
        // -0 must sort before +0
        printf("  -0 before +0: ");
        std::vector<float> z = {0.0f,-0.0f,0.0f,-0.0f};
        radix_sort(z.data(), z.size());
        bool zok = std::signbit(z[0])&&std::signbit(z[1])&&!std::signbit(z[2])&&!std::signbit(z[3]);
        printf("%s\n", zok?"OK":"FAIL"); if(!zok) failures++;
    }
    printf(failures? "RADIX FAILURES=%d\n":"ALL RADIX TESTS PASS\n", failures);
    return failures?1:0;
}
