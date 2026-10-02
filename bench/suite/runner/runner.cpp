// fyx benchmark suite orchestrator.
//
// Built by start.sh / start.cmd with whatever C++17 compiler is found, then:
//   1. detects the environment (OS, CPU, ISA, threads, RAM, compilers, rustc);
//   2. runs fyx's own correctness tests (native + portable builds);
//   3. builds one executable per competitor and configuration;
//   4. runs the matrix in interleaved rounds (algorithm order rotates per round),
//      with resumable per-(suite,type,algorithm,round) result files;
//   5. aggregates medians and writes report.md / cases.csv / summary.csv;
//   6. packs results into results_<stamp>.tar.gz for sending back.
//
// Fairness rules encoded here:
//   * every competitor is compiled with the most favourable portable recipe we
//     know (-O3, host-native ISA, its parallel backend when it has one);
//   * fyx is built twice: native, and "portable" (no ISA flags) -- the latter
//     is its least favourable configuration and is compared against the
//     competitors' most favourable one;
//   * nothing is disabled to make the comparison easier; a competitor that
//     cannot be built is reported as unavailable together with the reason.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define popen _popen
#  define pclose _pclose
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define FB_X86 1
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#else
#  define FB_X86 0
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------
static std::mutex g_print_mu;
static void say(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_print_mu);
    std::cout << s << std::endl;
}
static std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }
static std::string q(const std::string& p) { return "\"" + p + "\""; }

static int run(const std::string& cmd) {
#if defined(_WIN32)
    const std::string wrapped = "\"" + cmd + "\"";
    return std::system(wrapped.c_str());
#else
    return std::system(cmd.c_str());
#endif
}
static int run_log(const std::string& cmd, const fs::path& log) {
    {
        std::ofstream f(log, std::ios::app);
        f << "$ " << cmd << "\n";
    }
    return run(cmd + " >> " + q(log) + " 2>&1");
}
static std::string capture(const std::string& cmd) {
    std::string out;
#if defined(_WIN32)
    const std::string full = "\"" + cmd + " 2>&1\"";
#else
    const std::string full = cmd + " 2>&1";
#endif
    FILE* p = popen(full.c_str(), "r");
    if (!p) return out;
    char buf[4096];
    while (std::fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    return out;
}
static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}
static std::string first_line(const std::string& s) {
    const auto p = s.find('\n');
    return trim(p == std::string::npos ? s : s.substr(0, p));
}
static std::vector<std::string> split(const std::string& s, char c = ',') {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, c)) if (!it.empty()) out.push_back(it);
    return out;
}
static std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string r;
    for (std::size_t i = 0; i < v.size(); ++i) r += (i ? sep : "") + v[i];
    return r;
}
static std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static std::string tail(const std::string& s, std::size_t lines) {
    std::size_t pos = s.size(), count = 0;
    while (pos > 0) {
        if (s[pos - 1] == '\n' && ++count > lines) break;
        --pos;
    }
    return s.substr(pos);
}
static std::string stamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof b, "%Y%m%d_%H%M%S", &tm);
    return b;
}
static bool file_done(const fs::path& p) {
    if (!fs::exists(p)) return false;
    const std::string s = read_file(p);
    return s.find("#done") != std::string::npos;
}
static std::string fmt(double v, int prec = 3) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
static std::string human_secs(double s) {
    char b[64];
    if (s < 1e-3) std::snprintf(b, sizeof b, "%.1fus", s * 1e6);
    else if (s < 1) std::snprintf(b, sizeof b, "%.2fms", s * 1e3);
    else std::snprintf(b, sizeof b, "%.3fs", s);
    return b;
}

// ---------------------------------------------------------------------------
// environment
// ---------------------------------------------------------------------------
struct Cpu {
    bool x86 = FB_X86;
    bool avx2 = false, avx512 = false;   // avx512 = F+BW+DQ+VL
    std::string brand;
};
static Cpu detect_cpu() {
    Cpu c;
#if FB_X86
    unsigned r[4] = {0, 0, 0, 0};
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#  if defined(_MSC_VER) && !defined(__clang__)
        int t[4];
        __cpuidex(t, static_cast<int>(leaf), static_cast<int>(sub));
        for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(t[i]);
#  else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
    };
    cpuid(0, 0);
    const unsigned maxleaf = r[0];
    cpuid(1, 0);
    const bool osxsave = (r[2] >> 27) & 1;
    unsigned long long xcr0 = 0;
    if (osxsave) {
#  if defined(_MSC_VER) && !defined(__clang__)
        xcr0 = _xgetbv(0);
#  else
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#  endif
    }
    if (maxleaf >= 7) {
        cpuid(7, 0);
        const bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xe6) == 0xe6;
        c.avx2 = ymm && ((r[1] >> 5) & 1);
        c.avx512 = zmm && ((r[1] >> 16) & 1) && ((r[1] >> 17) & 1) && ((r[1] >> 30) & 1) && ((r[1] >> 31) & 1);
    }
    char brand[49] = {0};
    cpuid(0x80000000u, 0);
    if (r[0] >= 0x80000004u) {
        for (unsigned i = 0; i < 3; ++i) {
            cpuid(0x80000002u + i, 0);
            std::memcpy(brand + i * 16, r, 16);
        }
        c.brand = trim(brand);
    }
#endif
    return c;
}

struct Toolchain {
    std::string cxx, cc;
    std::string family;          // gnu | msvc
    std::string version;
    bool windows = false;
    std::string exe_ext;
    std::string base_cxx, base_c;
    std::string native_flags;    // host ISA flags
    std::string portable_flags;  // deliberately none
    std::string thread_flags;
    std::string rustc;           // empty = unavailable
    std::string rust_libs;
    std::string rust_target_flags;
};

struct Paths {
    fs::path root, fyx, suite, third, build, results;
};

static bool probe_compile(const Toolchain& t, const Paths& P, const std::string& flags,
                          const std::string& extra_src = "", bool run_it = true) {
    const fs::path dir = P.build / "probe";
    fs::create_directories(dir);
    static std::atomic<int> counter{0};
    const int id = counter++;
    const fs::path src = dir / ("p" + std::to_string(id) + ".cpp");
    const fs::path exe = dir / ("p" + std::to_string(id) + t.exe_ext);
    {
        std::ofstream f(src);
        f << (extra_src.empty() ? "#include <thread>\nint main(){ std::thread th([]{}); th.join(); return 0; }\n" : extra_src);
    }
    std::string cmd;
    if (t.family == "msvc")
        cmd = q(t.cxx) + " " + t.base_cxx + " " + flags + " " + q(src) + " /Fo" + q((dir / "").string()) + " /Fe" + q(exe);
    else
        cmd = q(t.cxx) + " " + t.base_cxx + " " + flags + " " + q(src) + " -o " + q(exe);
    const fs::path log = dir / "probe.log";
    if (run_log(cmd, log) != 0 || !fs::exists(exe)) return false;
    if (!run_it) return true;
    return run_log(q(exe), log) == 0;
}

static Toolchain detect_toolchain(const Paths& P, const std::string& want_cxx, const std::string& want_cc,
                                  const Cpu& cpu) {
    Toolchain t;
#if defined(_WIN32)
    t.windows = true;
    t.exe_ext = ".exe";
#endif
    std::vector<std::string> cands;
    if (!want_cxx.empty()) cands.push_back(want_cxx);
    if (const char* e = std::getenv("CXX")) cands.push_back(e);
#if defined(_WIN32)
    cands.insert(cands.end(), {"g++", "clang++", "cl"});
#else
    cands.insert(cands.end(), {"g++", "clang++", "c++"});
#endif
    for (const auto& c : cands) {
        std::string v = capture(q(c) + (c == "cl" || c.find("cl.exe") != std::string::npos ? "" : " --version"));
        if (v.find("Microsoft") != std::string::npos && v.find("C/C++") != std::string::npos) {
            t.cxx = c; t.family = "msvc"; t.version = first_line(v);
            break;
        }
        if (v.find("clang") != std::string::npos || v.find("Free Software") != std::string::npos ||
            v.find("g++") != std::string::npos || v.find("GCC") != std::string::npos) {
            t.cxx = c; t.family = "gnu"; t.version = first_line(v);
            break;
        }
    }
    if (t.cxx.empty()) return t;
    if (t.family == "msvc") {
        t.cc = t.cxx;
        t.base_cxx = "/nologo /std:c++17 /O2 /Oi /DNDEBUG /EHsc /permissive- /Zc:__cplusplus /utf-8 /bigobj";
        t.base_c = "/nologo /O2 /Oi /DNDEBUG /TC /utf-8";
        if (cpu.avx512 && probe_compile(t, P, "/arch:AVX512")) t.native_flags = "/arch:AVX512";
        else if (cpu.avx2 && probe_compile(t, P, "/arch:AVX2")) t.native_flags = "/arch:AVX2";
        return t;
    }
    // gnu-like: derive the C compiler
    if (!want_cc.empty()) t.cc = want_cc;
    else if (const char* e = std::getenv("CC")) t.cc = e;
    else {
        std::string c = t.cxx;
        const auto pos = c.rfind("++");
        if (pos != std::string::npos) {
            if (c.compare(pos - 1, 3, "g++") == 0) c.replace(pos - 1, 3, "gcc");
            else c.replace(pos, 2, "");
            if (c == "c") c = "cc";
        }
        t.cc = c;
    }
    t.base_cxx = "-std=c++17 -O3 -DNDEBUG";
    t.base_c = "-std=gnu11 -O3 -DNDEBUG";
    if (probe_compile(t, P, "-pthread")) t.thread_flags = "-pthread";
    for (const char* f : {"-march=native", "-mcpu=native"}) {
        if (probe_compile(t, P, std::string(f) + " " + t.thread_flags)) { t.native_flags = f; break; }
    }
    return t;
}

static void detect_rust(Toolchain& t, const Paths& P) {
    std::string v = capture("rustc --version");
    if (v.rfind("rustc ", 0) != 0) return;
    t.rustc = "rustc";
    t.rust_target_flags = "-C opt-level=3 -C panic=abort -C codegen-units=1";
    if (!t.native_flags.empty()) t.rust_target_flags += " -C target-cpu=native";
    const fs::path dir = P.build / "rust";
    fs::create_directories(dir);
    const std::string out = capture("rustc --edition 2021 --crate-type staticlib --print native-static-libs " +
                                    q(P.suite / "algos" / "rust_sorts.rs") + " -o " + q(dir / "probe_lib"));
    const auto k = out.find("native-static-libs:");
    if (k != std::string::npos) t.rust_libs = first_line(out.substr(k + 19));
}

static std::string env_report(const Toolchain& t, const Cpu& cpu) {
    std::ostringstream o;
    o << "date: " << stamp() << "\n";
    o << "hardware_concurrency: " << std::thread::hardware_concurrency() << "\n";
    o << "cpu_brand(cpuid): " << (cpu.brand.empty() ? "(n/a)" : cpu.brand) << "\n";
    o << "x86: " << cpu.x86 << " avx2: " << cpu.avx2 << " avx512(F/BW/DQ/VL): " << cpu.avx512 << "\n";
    o << "cxx: " << t.cxx << " [" << t.family << "] " << t.version << "\n";
    o << "cc: " << t.cc << "\n";
    o << "native_flags: '" << t.native_flags << "'  portable_flags: '" << t.portable_flags
      << "'  thread_flags: '" << t.thread_flags << "'\n";
    o << "rustc: " << (t.rustc.empty() ? "(not found)" : first_line(capture("rustc --version"))) << "\n";
#if defined(_WIN32)
    o << "\n--- os ---\n" << capture("ver");
    o << capture("powershell -NoProfile -Command \"Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | Format-List; Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Format-List\"");
#elif defined(__APPLE__)
    o << "\n--- os ---\n" << capture("sw_vers") << capture("uname -a");
    o << capture("sysctl -n machdep.cpu.brand_string hw.ncpu hw.physicalcpu hw.memsize hw.l2cachesize hw.l3cachesize");
#else
    o << "\n--- os ---\n" << capture("uname -a") << capture("cat /etc/os-release 2>/dev/null | head -4");
    o << "\n--- lscpu ---\n" << capture("lscpu 2>/dev/null");
    o << "\n--- memory ---\n" << capture("grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null");
    o << "\n--- governor ---\n" << capture("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null");
    o << "\n--- container ---\n" << capture("cat /proc/1/cgroup 2>/dev/null | head -3");
#endif
    return o.str();
}

static unsigned long long total_ram_bytes() {
#if defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string k;
    unsigned long long v;
    std::string unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return v * 1024ull;
#endif
    return 0;
}

// ---------------------------------------------------------------------------
// build specs
// ---------------------------------------------------------------------------
struct Spec {
    std::string id;            // algo@config, unique
    std::string src;           // algos/<file>
    std::vector<std::string> defines;
    std::string config;        // native | portable
    std::vector<fs::path> incs;
    std::vector<std::string> objs_key;   // prebuilt object groups (hwy, scandum0, ...)
    std::vector<std::string> lib_variants{""};   // tried in order
    std::string requires_;     // "x86", "rust", "pthread"
    std::string extra_flags;   // raw compiler flags (e.g. -U_REENTRANT)
    // filled after build:
    bool ok = false;
    std::string reason;
    fs::path exe;
    std::string name;
    bool parallel = false, stable = false;
    int suites = 0;
    std::set<std::string> types;
};

struct ObjGroup {
    std::vector<fs::path> objs;
    bool ok = false;
    std::string reason;
};

static std::string cfg_flags(const Toolchain& t, const std::string& cfg) {
    std::string f = cfg == "native" ? t.native_flags : t.portable_flags;
    if (!t.thread_flags.empty()) f += " " + t.thread_flags;
    return f;
}

static bool compile_obj(const Toolchain& t, const fs::path& src, const fs::path& obj, const std::string& flags,
                        const std::vector<fs::path>& incs, bool is_c, const fs::path& log) {
    if (fs::exists(obj) && fs::last_write_time(obj) >= fs::last_write_time(src)) return true;
    fs::create_directories(obj.parent_path());
    std::string cmd;
    std::string inc;
    for (const auto& i : incs) inc += (t.family == "msvc" ? " /I" : " -I") + q(i);
    if (t.family == "msvc")
        cmd = q(t.cxx) + " " + (is_c ? t.base_c : t.base_cxx) + " " + flags + inc + " /c " + q(src) + " /Fo" + q(obj);
    else
        cmd = q(is_c ? t.cc : t.cxx) + " " + (is_c ? t.base_c : t.base_cxx) + " " + flags + inc + " -c " + q(src) + " -o " + q(obj);
    return run_log(cmd, log) == 0 && fs::exists(obj);
}

static std::string msvc_defs(const Toolchain& t, const std::vector<std::string>& d) {
    std::string r;
    for (auto& x : d) r += (t.family == "msvc" ? " /D" : " -D") + x;
    return r;
}

template <class F>
static void parallel_for(std::size_t count, unsigned jobs, F&& f) {
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> th;
    for (unsigned j = 0; j < std::max(1u, jobs); ++j)
        th.emplace_back([&] {
            for (std::size_t i; (i = next++) < count;) f(i);
        });
    for (auto& x : th) x.join();
}

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------
struct Row {
    std::string suite, type, dist, algo, cfg;
    std::size_t n = 0;
    int round = 0;
    double secs = 0;
    bool ok = true;
};
static std::vector<Row> parse_csv(const fs::path& p) {
    std::vector<Row> rows;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line);
        if (c.size() < 9) continue;
        Row r;
        r.suite = c[0]; r.type = c[1]; r.dist = c[2]; r.n = std::stoull(c[3]);
        r.algo = c[4]; r.cfg = c[5]; r.round = std::atoi(c[6].c_str());
        r.secs = std::atof(c[7].c_str()); r.ok = c[8] == "1";
        rows.push_back(r);
    }
    return rows;
}
static double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Options {
    std::string profile = "full";
    std::vector<std::size_t> sizes;
    int rounds = 0;
    std::vector<std::string> only_algos, skip_algos, types, dists, suites;
    std::string cxx, cc;
    unsigned jobs = 0;
    bool skip_tests = false, resume = false, no_pack = false, build_only = false;
    std::uint64_t seed = 20261002;
    std::string out;
};

static void usage() {
    std::cout <<
        "runner [options]\n"
        "  --profile=quick|full|huge   (default full)\n"
        "  --sizes=1000,1000000        override sizes\n"
        "  --rounds=N                  override rounds\n"
        "  --algos=a,b / --skip=a,b    filter by algorithm name (e.g. fyx,ips4o_par)\n"
        "  --types=i32,str / --dists=random,zipf / --suites=unstable,stable\n"
        "  --cxx=g++ --cc=gcc          choose compilers (default: auto detect)\n"
        "  --jobs=N                    parallel build jobs\n"
        "  --skip-tests --build-only --no-pack --resume --seed=N\n";
}

int main(int argc, char** argv) {
    Options o;
    Paths P;
    P.root = fs::current_path();
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&](const char* k) -> const char* {
            const std::size_t n = std::strlen(k);
            return s.compare(0, n, k) == 0 ? s.c_str() + n : nullptr;
        };
        if (const char* v = val("--profile=")) o.profile = v;
        else if (const char* v1 = val("--sizes=")) for (auto& x : split(v1)) o.sizes.push_back(std::stoull(x));
        else if (const char* v2 = val("--rounds=")) o.rounds = std::atoi(v2);
        else if (const char* v3 = val("--algos=")) o.only_algos = split(v3);
        else if (const char* v4 = val("--skip=")) o.skip_algos = split(v4);
        else if (const char* v5 = val("--types=")) o.types = split(v5);
        else if (const char* v6 = val("--dists=")) o.dists = split(v6);
        else if (const char* v7 = val("--suites=")) o.suites = split(v7);
        else if (const char* v8 = val("--cxx=")) o.cxx = v8;
        else if (const char* v9 = val("--cc=")) o.cc = v9;
        else if (const char* v10 = val("--jobs=")) o.jobs = static_cast<unsigned>(std::atoi(v10));
        else if (const char* v11 = val("--root=")) P.root = v11;
        else if (const char* v12 = val("--seed=")) o.seed = std::stoull(v12);
        else if (const char* v13 = val("--out=")) o.out = v13;
        else if (s == "--skip-tests") o.skip_tests = true;
        else if (s == "--resume") o.resume = true;
        else if (s == "--no-pack") o.no_pack = true;
        else if (s == "--build-only") o.build_only = true;
        else if (s == "--help" || s == "-h") { usage(); return 0; }
        else { std::cerr << "unknown option " << s << "\n"; usage(); return 2; }
    }
    P.root = fs::absolute(P.root);
    P.fyx = P.root / "fyx";
    P.suite = P.root / "suite";
    P.third = P.root / "third_party";
    P.results = P.root / "results";
    if (!fs::exists(P.fyx / "fyx_sort.hpp")) {
        std::cerr << "fyx/fyx_sort.hpp not found under " << P.root << " (run update first)\n";
        return 2;
    }

    // ---- profile -----------------------------------------------------------
    std::size_t max_str = 1000000, max_kv = 10000000;
    if (o.profile == "quick") {
        if (o.sizes.empty()) o.sizes = {10000, 1000000};
        if (!o.rounds) o.rounds = 3;
    } else if (o.profile == "huge") {
        if (o.sizes.empty()) o.sizes = {1000, 10000, 100000, 1000000, 10000000, 100000000};
        if (!o.rounds) o.rounds = 5;
    } else {
        if (o.sizes.empty()) o.sizes = {1000, 10000, 100000, 1000000, 10000000};
        if (!o.rounds) o.rounds = 5;
    }
    if (!o.jobs) o.jobs = std::max(1u, std::thread::hardware_concurrency());

    // ---- environment ------------------------------------------------------
    const Cpu cpu = detect_cpu();
    say("== fyx benchmark suite ==");
    say("root: " + P.root.string());
    {
        // build dir is per compiler so switching compilers never mixes objects
        P.build = P.root / "build" / "tmp";
        fs::create_directories(P.build);
    }
    Toolchain tc = detect_toolchain(P, o.cxx, o.cc, cpu);
    if (tc.cxx.empty()) {
        std::cerr << "no C++17 compiler found (tried g++, clang++, cl)\n";
        return 3;
    }
    std::string tag = tc.family + "_" + fs::path(tc.cxx).stem().string();
    P.build = P.root / "build" / tag;
    fs::create_directories(P.build / "bin");
    fs::create_directories(P.build / "logs");
    detect_rust(tc, P);

    fs::path out_dir;
    if (!o.out.empty()) out_dir = fs::absolute(o.out);
    else if (o.resume && fs::exists(P.results / "LATEST")) out_dir = P.results / trim(read_file(P.results / "LATEST"));
    else out_dir = P.results / (stamp() + "_" + tag);
    fs::create_directories(out_dir / "raw");
    fs::create_directories(out_dir / "logs");
    {
        std::ofstream f(P.results / "LATEST");
        f << out_dir.filename().string() << "\n";
    }
    const std::string env = env_report(tc, cpu);
    {
        std::ofstream f(out_dir / "env.txt");
        f << env;
    }
    say("compiler: " + tc.cxx + " (" + tc.version + ")");
    say("native flags: '" + tc.native_flags + "', threads: " + std::to_string(std::thread::hardware_concurrency()) +
        ", rustc: " + (tc.rustc.empty() ? "no" : "yes"));
    say("output: " + out_dir.string());

    std::ofstream summary_md;   // opened at the end
    std::vector<std::string> test_lines;

    // ---- fyx correctness tests -------------------------------------------
    if (!o.skip_tests && fs::exists(P.fyx / "test")) {
        say("\n== fyx correctness tests ==");
        std::vector<fs::path> tests;
        for (auto& e : fs::directory_iterator(P.fyx / "test"))
            if (e.path().extension() == ".cpp") tests.push_back(e.path());
        std::sort(tests.begin(), tests.end());
        struct TJob { fs::path src; std::string cfg; };
        std::vector<TJob> jobs;
        for (auto& t : tests) jobs.push_back({t, "native"});
        for (auto& t : tests) if (t.stem() == "t_api" || t.stem() == "t_adaptive") jobs.push_back({t, "portable"});
        std::vector<std::string> res(jobs.size());
        parallel_for(jobs.size(), std::max(1u, o.jobs / 2), [&](std::size_t i) {
            const auto& j = jobs[i];
            const std::string id = j.src.stem().string() + "@" + j.cfg;
            const fs::path exe = P.build / "bin" / ("test_" + j.src.stem().string() + "_" + j.cfg + tc.exe_ext);
            const fs::path log = out_dir / "logs" / ("test_" + j.src.stem().string() + "_" + j.cfg + ".log");
            std::string flags = cfg_flags(tc, j.cfg);
            fs::create_directories(P.build / "obj" / ("test_" + j.cfg));
            std::string cmd = tc.family == "msvc"
                ? q(tc.cxx) + " " + tc.base_cxx + " " + flags + " " + q(j.src) + " /Fo" + q((P.build / "obj" / ("test_" + j.cfg) / "").string()) + " /Fe" + q(exe)
                : q(tc.cxx) + " " + tc.base_cxx + " " + flags + " " + q(j.src) + " -o " + q(exe) + " -latomic";
            int rc = run_log(cmd, log);
            if (rc != 0 && tc.family != "msvc") {
                cmd = cmd.substr(0, cmd.size() - std::string(" -latomic").size());
                rc = run_log(cmd, log);
            }
            if (rc != 0) { res[i] = "| " + id + " | BUILD FAIL | see logs/" + log.filename().string() + " |"; say("  " + id + ": BUILD FAIL"); return; }
            const auto t0 = std::chrono::steady_clock::now();
            rc = run_log(q(exe), log);
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            res[i] = "| " + id + " | " + (rc == 0 ? "PASS" : "FAIL rc=" + std::to_string(rc)) + " | " + fmt(el, 1) + "s |";
            say("  " + id + ": " + (rc == 0 ? "PASS" : "FAIL") + " (" + fmt(el, 1) + "s)");
        });
        test_lines = res;
    }

    // ---- object groups (highway, scandum) ---------------------------------
    std::map<std::string, ObjGroup> groups;
    const fs::path hwy = P.third / "highway";
    const fs::path boost_inc = [&] {
        if (const char* b = std::getenv("BOOST_ROOT")) return fs::path(b);
        return P.third / "boost";
    }();
    say("\n== building competitors ==");
    {
        ObjGroup g;
        if (!fs::exists(hwy / "hwy" / "contrib" / "sort" / "vqsort.h")) {
            g.reason = "third_party/highway missing";
        } else {
            std::vector<fs::path> srcs;
            for (const char* f : {"abort.cc", "aligned_allocator.cc", "nanobenchmark.cc", "per_target.cc", "print.cc", "targets.cc", "timer.cc"})
                srcs.push_back(hwy / "hwy" / f);
            for (auto& e : fs::directory_iterator(hwy / "hwy" / "contrib" / "sort")) {
                const std::string fn = e.path().filename().string();
                if (fn.rfind("vqsort", 0) == 0 && e.path().extension() == ".cc") srcs.push_back(e.path());
            }
            std::vector<char> okv(srcs.size(), 0);
            const fs::path log = out_dir / "logs" / "build_highway.log";
            say("  compiling highway/vqsort (" + std::to_string(srcs.size()) + " files, cached after first run)...");
            parallel_for(srcs.size(), o.jobs, [&](std::size_t i) {
                const fs::path obj = P.build / "obj" / "hwy" / (srcs[i].stem().string() + (tc.family == "msvc" ? ".obj" : ".o"));
                okv[i] = compile_obj(tc, srcs[i], obj, cfg_flags(tc, "native"), {hwy}, false, log);
            });
            for (std::size_t i = 0; i < srcs.size(); ++i)
                g.objs.push_back(P.build / "obj" / "hwy" / (srcs[i].stem().string() + (tc.family == "msvc" ? ".obj" : ".o")));
            g.ok = std::all_of(okv.begin(), okv.end(), [](char c) { return c != 0; });
            if (!g.ok) g.reason = "highway build failed: " + tail(read_file(log), 6);
        }
        groups["hwy"] = g;
    }
    for (int m = 0; m < 3; ++m) {
        ObjGroup g;
        static const char* dirs[] = {"crumsort", "fluxsort", "quadsort"};
        const fs::path inc = P.third / dirs[m];
        const fs::path obj = P.build / "obj" / ("scandum" + std::to_string(m) + (tc.family == "msvc" ? ".obj" : ".o"));
        const fs::path log = out_dir / "logs" / ("build_scandum" + std::to_string(m) + ".log");
        // scandum sorts move records through integer-typed pointers (type
        // punning); without -fno-strict-aliasing GCC -O3 miscompiles the
        // struct-with-comparator path (observed: unsorted quadsort output).
        const std::string def = tc.family == "msvc" ? " /DFB_SCANDUM=" + std::to_string(m)
                                                    : " -fno-strict-aliasing -DFB_SCANDUM=" + std::to_string(m);
        if (fs::exists(obj)) fs::remove(obj);
        g.ok = fs::exists(inc) && compile_obj(tc, P.suite / "algos" / "scandum.c", obj, cfg_flags(tc, "native") + def, {inc}, true, log);
        if (g.ok) g.objs.push_back(obj);
        else g.reason = fs::exists(inc) ? "C build failed: " + tail(read_file(log), 6) : "third_party missing";
        groups["scandum" + std::to_string(m)] = g;
    }
    {
        // oneTBB, compiled straight from source into objects: IPS4o / IPS2Ra
        // parallel schedulers and libstdc++'s std::execution backend need it.
        ObjGroup g;
        const fs::path tbb = P.third / "oneTBB";
        if (!fs::exists(tbb / "src" / "tbb")) {
            g.reason = "third_party/oneTBB missing";
        } else {
            std::vector<fs::path> srcs;
            for (auto& e : fs::directory_iterator(tbb / "src" / "tbb"))
                if (e.path().extension() == ".cpp") srcs.push_back(e.path());
            std::sort(srcs.begin(), srcs.end());
            std::vector<char> okv(srcs.size(), 0);
            const fs::path log = out_dir / "logs" / "build_onetbb.log";
            const std::string defs = msvc_defs(tc, {"__TBB_BUILD=1", "__TBB_DYNAMIC_LOAD_ENABLED=0",
                                                    "__TBB_SOURCE_DIRECTLY_INCLUDED=1", "TBB_USE_ASSERT=0"});
            say("  compiling oneTBB (" + std::to_string(srcs.size()) + " files, cached after first run)...");
            parallel_for(srcs.size(), o.jobs, [&](std::size_t i) {
                const fs::path obj = P.build / "obj" / "tbb" / (srcs[i].stem().string() + (tc.family == "msvc" ? ".obj" : ".o"));
                okv[i] = compile_obj(tc, srcs[i], obj, cfg_flags(tc, "native") + defs, {tbb / "include", tbb / "src"}, false, log);
            });
            for (auto& sp : srcs)
                g.objs.push_back(P.build / "obj" / "tbb" / (sp.stem().string() + (tc.family == "msvc" ? ".obj" : ".o")));
            g.ok = std::all_of(okv.begin(), okv.end(), [](char c) { return c != 0; });
            if (!g.ok) g.reason = "oneTBB build failed: " + tail(read_file(log), 6);
        }
        groups["tbb"] = g;
    }
    if (!tc.rustc.empty()) {
        ObjGroup g;
        const fs::path lib = P.build / "obj" / (tc.family == "msvc" ? "rust_sorts.lib" : "librust_sorts.a");
        fs::create_directories(lib.parent_path());
        const fs::path log = out_dir / "logs" / "build_rust.log";
        g.ok = run_log("rustc --edition 2021 --crate-type staticlib " + tc.rust_target_flags + " " +
                       q(P.suite / "algos" / "rust_sorts.rs") + " -o " + q(lib), log) == 0 && fs::exists(lib);
        if (g.ok) g.objs.push_back(lib);
        else g.reason = "rustc failed: " + tail(read_file(log), 6);
        groups["rust"] = g;
    } else {
        groups["rust"] = ObjGroup{{}, false, "rustc not found (install Rust to include Rust std sort_unstable/sort)"};
    }

    // ---- specs ------------------------------------------------------------
    std::vector<Spec> specs;
    auto add = [&](std::string id, std::string src, std::vector<std::string> defs, std::string cfg,
                   std::vector<fs::path> incs, std::vector<std::string> objs = {},
                   std::vector<std::string> libs = {""}, std::string req = "") {
        Spec s;
        s.id = id; s.src = src; s.defines = defs; s.config = cfg; s.incs = incs;
        s.objs_key = objs; s.lib_variants = libs; s.requires_ = req;
        specs.push_back(s);
    };
    const bool msvc = tc.family == "msvc";
    const std::vector<std::string> atomic_libs = msvc ? std::vector<std::string>{""} : std::vector<std::string>{"", "-latomic"};
    const std::vector<std::string> tbb_libs = msvc ? std::vector<std::string>{""} : std::vector<std::string>{"-ldl", "-ldl -latomic", ""};
    const std::vector<std::string> tbb_libs_dl = tbb_libs;
    for (const char* cfg : {"native", "portable"}) {
        add(std::string("fyx@") + cfg, "fyx.cpp", {"FB_FYX_MODE=0"}, cfg, {P.fyx}, {}, atomic_libs);
        add(std::string("fyx_par@") + cfg, "fyx.cpp", {"FB_FYX_MODE=1"}, cfg, {P.fyx}, {}, atomic_libs);
        add(std::string("fyx_stable@") + cfg, "fyx.cpp", {"FB_FYX_MODE=2"}, cfg, {P.fyx}, {}, atomic_libs);
    }
    add("std_sort@native", "std_sort.cpp", {"FB_STD_MODE=0"}, "native", {});
    add("std_stable@native", "std_sort.cpp", {"FB_STD_MODE=1"}, "native", {});
    const fs::path tbb_inc = P.third / "oneTBB" / "include";
    // libstdc++ silently falls back to a SERIAL backend when TBB headers are
    // absent, so the GNU build links the vendored oneTBB explicitly.
    add("std_par@native", "std_sort.cpp", {"FB_STD_MODE=2"}, "native", msvc ? std::vector<fs::path>{} : std::vector<fs::path>{tbb_inc},
        msvc ? std::vector<std::string>{} : std::vector<std::string>{"tbb"}, tbb_libs);
    add("std_par_stable@native", "std_sort.cpp", {"FB_STD_MODE=3"}, "native", msvc ? std::vector<fs::path>{} : std::vector<fs::path>{tbb_inc},
        msvc ? std::vector<std::string>{} : std::vector<std::string>{"tbb"}, tbb_libs);
    add("pdqsort@native", "pdqsort.cpp", {}, "native", {P.third / "pdqsort"});
    add("ips4o@native", "ips4o.cpp", {"FB_PAR=0"}, "native", {P.third / "ips4o" / "include", tbb_inc}, {"tbb"}, tbb_libs_dl);
    add("ips4o_par@native", "ips4o.cpp", {"FB_PAR=1"}, "native", {P.third / "ips4o" / "include", tbb_inc}, {"tbb"}, tbb_libs_dl, "pthread");
    add("ips2ra@native", "ips2ra.cpp", {"FB_PAR=0"}, "native", {P.third / "ips2ra" / "include", tbb_inc}, {"tbb"}, tbb_libs_dl);
    add("ips2ra_par@native", "ips2ra.cpp", {"FB_PAR=1"}, "native", {P.third / "ips2ra" / "include", tbb_inc}, {"tbb"}, tbb_libs_dl, "pthread");
    // (IPS4o/IPS2Ra include their TBB-based scheduler header even for the
    //  sequential entry point, so both variants get the oneTBB headers/objects.)
    add("vqsort@native", "vqsort.cpp", {}, "native", {hwy}, {"hwy"});
    add("xss@native", "xss.cpp", {}, "native", {P.third / "x86-simd-sort" / "src"}, {}, {""}, "x86");
    add("ska_sort@native", "ska_sort.cpp", {}, "native", {P.third / "ska_sort"});
    for (int m = 0; m <= 6; ++m) {
        static const char* nm[] = {"boost_spreadsort", "boost_block_indirect", "boost_sample_sort",
                                   "boost_parallel_stable", "boost_spinsort", "boost_flat_stable", "boost_pdqsort"};
        add(std::string(nm[m]) + "@native", "boost_sort.cpp", {"FB_BOOST_MODE=" + std::to_string(m)}, "native",
            {boost_inc}, {}, atomic_libs);
    }
    add("crumsort@native", "scandum.cpp", {"FB_SCANDUM=0"}, "native", {}, {"scandum0"});
    add("fluxsort@native", "scandum.cpp", {"FB_SCANDUM=1"}, "native", {}, {"scandum1"});
    add("quadsort@native", "scandum.cpp", {"FB_SCANDUM=2"}, "native", {}, {"scandum2"});
    add("rust_unstable@native", "rust.cpp", {"FB_RUST_STABLE=0"}, "native", {}, {"rust"});
    add("rust_stable@native", "rust.cpp", {"FB_RUST_STABLE=1"}, "native", {}, {"rust"});

    auto algo_of = [](const std::string& id) { return id.substr(0, id.find('@')); };
    specs.erase(std::remove_if(specs.begin(), specs.end(), [&](const Spec& s) {
        const std::string a = algo_of(s.id);
        if (!o.only_algos.empty() && std::find(o.only_algos.begin(), o.only_algos.end(), a) == o.only_algos.end()) return true;
        return std::find(o.skip_algos.begin(), o.skip_algos.end(), a) != o.skip_algos.end();
    }), specs.end());

    parallel_for(specs.size(), o.jobs, [&](std::size_t i) {
        Spec& s = specs[i];
        if (s.requires_ == "x86" && !cpu.x86) { s.reason = "x86-only library; host is not x86"; return; }
        if (s.requires_ == "pthread" && msvc) { /* try anyway; may fail */ }
        for (const auto& k : s.objs_key) {
            if (!groups[k].ok) { s.reason = groups[k].reason; return; }
        }
        std::string safe = s.id;
        std::replace(safe.begin(), safe.end(), '@', '_');
        s.exe = P.build / "bin" / (safe + tc.exe_ext);
        const fs::path log = out_dir / "logs" / ("build_" + safe + ".log");
        std::string defs, incs;
        for (auto& d : s.defines) defs += (msvc ? " /D" : " -D") + d;
        if (!s.extra_flags.empty()) defs += " " + s.extra_flags;
        if (s.config == "portable") defs += msvc ? " /DFB_CONFIG_PORTABLE=1" : " -DFB_CONFIG_PORTABLE=1";
        for (auto& inc : s.incs) incs += (msvc ? " /I" : " -I") + q(inc);
        std::string objs;
        for (auto& k : s.objs_key) for (auto& ob : groups[k].objs) objs += " " + q(ob);
        if (s.objs_key.size() && s.objs_key[0] == "rust") objs += " ";
        bool built = false;
        for (const auto& libs : s.lib_variants) {
            std::string extra = libs;
            if (!s.objs_key.empty() && s.objs_key[0] == "rust") extra += " " + tc.rust_libs;
            std::string cmd;
            if (msvc) {
                const fs::path od = P.build / "obj" / safe;
                fs::create_directories(od);
                cmd = q(tc.cxx) + " " + tc.base_cxx + " " + cfg_flags(tc, s.config) + defs + incs + " " +
                      q(P.suite / "algos" / s.src) + objs + " /Fo" + q((od / "").string()) + " /Fe" + q(s.exe) +
                      (extra.empty() ? "" : " /link " + extra);
            } else {
                cmd = q(tc.cxx) + " " + tc.base_cxx + " " + cfg_flags(tc, s.config) + defs + incs + " " +
                      q(P.suite / "algos" / s.src) + objs + " -o " + q(s.exe) + " " + extra;
            }
            if (run_log(cmd, log) == 0 && fs::exists(s.exe)) { built = true; break; }
        }
        if (!built) { s.reason = "compile failed: " + tail(read_file(log), 8); return; }
        // query capabilities
        const std::string info = capture(q(s.exe) + " --info");
        for (auto& line : split(info, '\n')) {
            const std::string l = trim(line);
            auto kv = [&](const char* k) -> const char* { const auto n = std::strlen(k); return l.compare(0, n, k) == 0 ? l.c_str() + n : nullptr; };
            if (const char* v = kv("name=")) s.name = v;
            else if (const char* v2 = kv("parallel=")) s.parallel = std::atoi(v2) != 0;
            else if (const char* v3 = kv("stable=")) s.stable = std::atoi(v3) != 0;
            else if (const char* v4 = kv("suites=")) s.suites = std::atoi(v4);
            else if (const char* v5 = kv("types=")) for (auto& t : split(v5)) s.types.insert(t);
        }
        if (s.name.empty()) { s.reason = "executable does not run (--info failed): " + tail(info, 6); return; }
        s.ok = true;
    });
    int built = 0;
    for (auto& s : specs) {
        say(std::string("  ") + (s.ok ? "[ok]   " : "[n/a]  ") + s.id + (s.ok ? "  types=" + join(std::vector<std::string>(s.types.begin(), s.types.end()), ",") : "  -- " + first_line(s.reason)));
        built += s.ok;
    }
    if (o.build_only) return 0;

    // ---- run --------------------------------------------------------------
    struct Unit { std::string suite, type; };
    std::vector<Unit> units;
    const std::vector<std::string> ut = {"i32", "u32", "i64", "u64", "f32", "f64", "str", "kv16"};
    const std::vector<std::string> st = {"rec8", "kv16", "i32", "f64", "str"};
    auto want = [](const std::vector<std::string>& v, const std::string& s) {
        return v.empty() || std::find(v.begin(), v.end(), s) != v.end();
    };
    for (auto& t : ut) if (want(o.suites, "unstable") && want(o.types, t)) units.push_back({"unstable", t});
    for (auto& t : st) if (want(o.suites, "stable") && want(o.types, t)) units.push_back({"stable", t});

    std::string sizes_arg;
    for (std::size_t i = 0; i < o.sizes.size(); ++i) sizes_arg += (i ? "," : "") + std::to_string(o.sizes[i]);
    {
        const unsigned long long ram = total_ram_bytes();
        if (ram && o.sizes.back() >= 100000000ull && ram < 8ull * 1000000000ull)
            say("WARNING: available RAM < 8GB; 1e8-element cases may swap. Consider --profile=full.");
    }
    std::vector<Spec*> live;
    for (auto& s : specs) if (s.ok) live.push_back(&s);

    std::size_t total = 0;
    for (auto* s : live)
        for (auto& u : units) {
            const bool suite_ok = (u.suite == "unstable" && (s->suites & 1)) || (u.suite == "stable" && (s->suites & 2) && s->stable);
            if (suite_ok && s->types.count(u.type)) total += static_cast<std::size_t>(o.rounds);
        }
    say("\n== running: " + std::to_string(live.size()) + " implementations, " + std::to_string(units.size()) +
        " (suite,type) units, " + std::to_string(o.rounds) + " rounds, sizes=" + sizes_arg + " ==");
    {
        std::ofstream f(out_dir / "config.txt");
        f << "profile=" << o.profile << "\nsizes=" << sizes_arg << "\nrounds=" << o.rounds << "\nseed=" << o.seed
          << "\nmax_str=" << max_str << "\nmax_kv=" << max_kv << "\ndists=" << join(o.dists, ",") << "\n";
    }
    std::size_t done = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 1; r <= o.rounds; ++r) {
        for (std::size_t ui = 0; ui < units.size(); ++ui) {
            const Unit& u = units[ui];
            std::vector<Spec*> order;
            for (auto* s : live) {
                const bool suite_ok = (u.suite == "unstable" && (s->suites & 1)) || (u.suite == "stable" && (s->suites & 2) && s->stable);
                if (suite_ok && s->types.count(u.type)) order.push_back(s);
            }
            if (order.empty()) continue;
            // rotate (and reverse on even rounds) so no implementation is always first/last
            std::rotate(order.begin(), order.begin() + static_cast<std::ptrdiff_t>((r + ui) % order.size()), order.end());
            if (r % 2 == 0) std::reverse(order.begin(), order.end());
            for (auto* s : order) {
                std::string safe = s->id;
                std::replace(safe.begin(), safe.end(), '@', '_');
                const fs::path out = out_dir / "raw" / (u.suite + "_" + u.type + "_" + safe + "_r" + std::to_string(r) + ".csv");
                ++done;
                if (file_done(out)) continue;
                const fs::path tmp = out.string() + ".part";
                std::string cmd = q(s->exe) + " --suites=" + u.suite + " --types=" + u.type + " --sizes=" + sizes_arg +
                                  " --round=" + std::to_string(r) + " --seed=" + std::to_string(o.seed) +
                                  " --max-str=" + std::to_string(max_str) + " --max-kv=" + std::to_string(max_kv) +
                                  (o.dists.empty() ? "" : " --dists=" + join(o.dists, ",")) + " --out=" + q(tmp);
                const auto t0 = std::chrono::steady_clock::now();
                const int rc = run_log(cmd, out_dir / "logs" / ("run_" + safe + ".log"));
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (rc == 0 && file_done(tmp)) fs::rename(tmp, out);
                const double all = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const double eta = done ? all / static_cast<double>(done) * static_cast<double>(total - done) : 0;
                say("  [r" + std::to_string(r) + "/" + std::to_string(o.rounds) + " " + std::to_string(done) + "/" +
                    std::to_string(total) + "] " + u.suite + "/" + u.type + " " + s->id + "  " + fmt(el, 1) + "s" +
                    (rc ? "  (FAILED rc=" + std::to_string(rc) + ")" : "") + "  ETA " + fmt(eta / 60.0, 1) + " min");
            }
        }
    }

    // ---- aggregate --------------------------------------------------------
    say("\n== aggregating ==");
    std::vector<Row> rows;
    for (auto& e : fs::directory_iterator(out_dir / "raw"))
        if (e.path().extension() == ".csv") {
            auto r = parse_csv(e.path());
            rows.insert(rows.end(), r.begin(), r.end());
        }
    std::map<std::string, const Spec*> by_impl;
    for (auto& s : specs) if (s.ok) by_impl[s.name + "@" + s.config] = &s;

    using CaseKey = std::tuple<std::string, std::string, std::size_t, std::string>;   // suite,type,n,dist
    std::map<CaseKey, std::map<std::string, std::vector<double>>> times;
    std::vector<std::string> failures;
    for (auto& r : rows) {
        const std::string impl = r.algo + "@" + r.cfg;
        if (!r.ok || r.secs < 0) {
            failures.push_back(r.suite + "/" + r.type + "/" + r.dist + "/n=" + std::to_string(r.n) + " " + impl + " round " + std::to_string(r.round));
            continue;
        }
        times[CaseKey{r.suite, r.type, r.n, r.dist}][impl].push_back(r.secs);
    }

    std::ofstream cases(out_dir / "cases.csv");
    cases << "suite,type,n,dist,impl,median_s,min_s,max_s,rounds\n";
    struct Cmp { std::string label; int wins = 0, losses = 0, losses3 = 0, cells = 0; std::vector<std::pair<double, std::string>> lose_list; };
    Cmp cA{"A 串行对等：fyx 串行(native) vs 最佳串行对手(native)"};
    Cmp cB{"B 全能力：fyx 最佳(native,串/并行) vs 最佳对手(任意,native)"};
    Cmp cC{"C 最不利：fyx 最佳(portable,无ISA标志) vs 最佳对手(任意,native)"};
    std::ofstream wide(out_dir / "summary.csv");
    wide << "suite,type,n,dist,fyx_ser_native,fyx_ser_portable,fyx_par_native,fyx_par_portable,best_serial_opp,best_serial_opp_s,best_opp,best_opp_s,A_ratio,B_ratio,C_ratio\n";
    std::map<std::string, std::pair<int, int>> per_opp_wins;   // opponent -> (cells where it beat fyx B, cells)

    for (auto& [key, impls] : times) {
        const auto& [suite, type, n, dist] = key;
        std::map<std::string, double> med;
        for (auto& [impl, v] : impls) {
            med[impl] = median(v);
            cases << suite << "," << type << "," << n << "," << dist << "," << impl << "," << med[impl] << ","
                  << *std::min_element(v.begin(), v.end()) << "," << *std::max_element(v.begin(), v.end()) << "," << v.size() << "\n";
        }
        auto get = [&](const std::string& k) { auto it = med.find(k); return it == med.end() ? NAN : it->second; };
        const std::string ser = suite == "stable" ? "fyx_stable" : "fyx";
        const double fsn = get(ser + "@native"), fsp = get(ser + "@portable");
        const double fpn = suite == "stable" ? NAN : get("fyx_par@native");
        const double fpp = suite == "stable" ? NAN : get("fyx_par@portable");
        double bso = INFINITY, bo = INFINITY;
        std::string bson = "-", bon = "-";
        for (auto& [impl, t] : med) {
            if (impl.rfind("fyx", 0) == 0) continue;
            auto it = by_impl.find(impl);
            const bool par = it != by_impl.end() && it->second->parallel;
            if (!par && t < bso) { bso = t; bson = impl; }
            if (t < bo) { bo = t; bon = impl; }
        }
        auto mn = [](double a, double b) { return std::isnan(a) ? b : std::isnan(b) ? a : std::min(a, b); };
        const double fbn = mn(fsn, fpn), fbp = mn(fsp, fpp);
        auto tally = [&](Cmp& c, double fyx, double opp, const std::string& oppn) -> double {
            if (std::isnan(fyx) || !std::isfinite(opp)) return NAN;
            const double ratio = opp / fyx;   // >1 = fyx faster
            ++c.cells;
            if (ratio >= 1.0) ++c.wins;
            else {
                ++c.losses;
                if (ratio < 0.97) ++c.losses3;
                c.lose_list.push_back({ratio, suite + " | " + type + " | " + std::to_string(n) + " | " + dist + " | " +
                                                  oppn + " | " + human_secs(opp) + " vs fyx " + human_secs(fyx) + " | " + fmt(ratio, 3) + "x"});
            }
            return ratio;
        };
        const double ra = tally(cA, fsn, bso, bson);
        const double rb = tally(cB, fbn, bo, bon);
        const double rc = tally(cC, fbp, bo, bon);
        for (auto& [impl, t] : med) {
            if (impl.rfind("fyx", 0) == 0 || std::isnan(fbn)) continue;
            auto& pw = per_opp_wins[impl];
            ++pw.second;
            if (t < fbn) ++pw.first;
        }
        wide << suite << "," << type << "," << n << "," << dist << "," << fsn << "," << fsp << "," << fpn << "," << fpp << ","
             << bson << "," << bso << "," << bon << "," << bo << "," << ra << "," << rb << "," << rc << "\n";
    }

    // ---- report -----------------------------------------------------------
    std::ofstream md(out_dir / "report.md");
    md << "# fyx 基准测试报告\n\n";
    md << "生成时间：" << stamp() << "  \n输出目录：`" << out_dir.filename().string() << "`\n\n";
    md << "## 环境\n\n```\n" << env.substr(0, env.find("\n--- os")) << "\n```\n完整环境见 `env.txt`。\n\n";
    md << "## 配置\n\nprofile=" << o.profile << "，sizes=" << sizes_arg << "，rounds=" << o.rounds << "（取中位数），seed=" << o.seed
       << "。每轮按 (suite,type) 交错运行全部实现且顺序轮换；输入复制与校验不计时；n 小于 2^18 时同一批多份副本连续排序后取平均；每轮内重复计时至多 5 次（每次重新复制输入，n·批量 ≥ 2^23 时 1 次）取中位数，再跨轮取中位数。\n\n";
    md << "## fyx 正确性测试\n\n";
    if (test_lines.empty()) md << "（跳过）\n\n";
    else { md << "| test | 结果 | 用时 |\n|---|---|---|\n"; for (auto& l : test_lines) md << l << "\n"; md << "\n"; }
    md << "## 参测实现\n\n| 实现 | 状态 | 并行 | 稳定 | 类型 / 原因 |\n|---|---|---|---|---|\n";
    for (auto& s : specs) {
        md << "| " << s.id << " | " << (s.ok ? "已构建" : "不可用") << " | " << (s.ok ? (s.parallel ? "是" : "否") : "-") << " | "
           << (s.ok ? (s.stable ? "是" : "否") : "-") << " | ";
        if (s.ok) md << join(std::vector<std::string>(s.types.begin(), s.types.end()), ",");
        else { std::string r = first_line(s.reason); std::replace(r.begin(), r.end(), '|', '/'); md << r; }
        md << " |\n";
    }
    md << "\n## 正确性失败（任何一项都必须优先修复）\n\n";
    if (failures.empty()) md << "无。\n\n";
    else { for (std::size_t i = 0; i < failures.size() && i < 300; ++i) md << "- " << failures[i] << "\n"; md << "\n"; }
    md << "## 胜负汇总（比值 = 对手时间 / fyx 时间，>1 表示 fyx 更快）\n\n| 对比 | 测试格 | fyx 胜 | fyx 负 | 负且差距>3% |\n|---|---:|---:|---:|---:|\n";
    for (Cmp* c : {&cA, &cB, &cC})
        md << "| " << c->label << " | " << c->cells << " | " << c->wins << " | " << c->losses << " | " << c->losses3 << " |\n";
    md << "\n### 各对手在 B 口径下快于 fyx 的格数\n\n| 对手 | 快于 fyx | 可比格 |\n|---|---:|---:|\n";
    for (auto& [impl, pw] : per_opp_wins) md << "| " << impl << " | " << pw.first << " | " << pw.second << " |\n";
    for (Cmp* c : {&cA, &cB, &cC}) {
        std::sort(c->lose_list.begin(), c->lose_list.end());
        md << "\n### 未制霸格：" << c->label << "（最差在前，最多 300 条）\n\n";
        if (c->lose_list.empty()) { md << "无。\n"; continue; }
        md << "| suite | type | n | dist | 最快对手 | 时间 | 比值 |\n|---|---|---:|---|---|---|---:|\n";
        for (std::size_t i = 0; i < c->lose_list.size() && i < 300; ++i) md << "| " << c->lose_list[i].second << " |\n";
    }
    md << "\n全部中位数见 `cases.csv`，逐格对比见 `summary.csv`，原始数据见 `raw/`，构建/运行日志见 `logs/`。\n";
    md.close();
    say("report: " + (out_dir / "report.md").string());
    say("A losses=" + std::to_string(cA.losses) + "/" + std::to_string(cA.cells) + "  B losses=" + std::to_string(cB.losses) + "/" +
        std::to_string(cB.cells) + "  C losses=" + std::to_string(cC.losses) + "/" + std::to_string(cC.cells) +
        "  correctness failures=" + std::to_string(failures.size()));

    if (!o.no_pack) {
        const fs::path tgz = P.root / ("results_" + out_dir.filename().string() + ".tar.gz");
        const int rc = run("tar -czf " + q(tgz) + " -C " + q(out_dir.parent_path()) + " " + q(out_dir.filename().string()));
        if (rc == 0) say("\n请把这个文件发回： " + tgz.string());
        else say("\n打包失败；请手动压缩并发回目录： " + out_dir.string());
    }
    return 0;
}
