#!/usr/bin/env python3
"""Write a timing-instrumented copy of fyx_sort.hpp.

The sandbox wipes anything outside the repository, including the scratch
directory we used to keep this in.  So instead of keeping a modified header
around, regenerate it: run this script, then compile radix_timer.cpp against
its output.

    python3 tools/dev/instrument.py [out_dir]        # default: /tmp/fyxinst

It splits every fyx::detail::radix_sort_impl into four timers -- the fused
histogram, the encode pass, the scatter passes and the decode pass -- which is
how the radix's cost was attributed (see NOTES.md).
"""
import os
import sys

TIMERS = """namespace fyx {
namespace detail {
struct RadixTimers {
    double hist = 0, enc = 0, scat = 0, dec = 0;
    int passes = 0;
    static RadixTimers& get() { static RadixTimers t; return t; }
    static void reset() { get() = RadixTimers(); }
};
static inline double rt_now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}"""


def sub_first(s, old, new):
    """Replace the first occurrence only; assert that there is one."""
    i = s.find(old)
    assert i >= 0, old[:70]
    return s[:i] + new + s[i + len(old):]


def sub_all(s, old, new):
    """Replace every occurrence; assert that there is at least one."""
    assert s.count(old) >= 1, old[:70]
    return s.replace(old, new)


def build(src_path, out_dir):
    s = open(src_path).read()

    s = sub_first(s, "#include <cstddef>", "#include <cstddef>\n#include <chrono>")
    s = sub_first(s, "namespace fyx {\nnamespace detail {", TIMERS)

    # -- 1. fused histogram -------------------------------------------------
    s = sub_all(s,
                """    RadixHistogram<Passes> hist;
    scalar_histogram<T, Passes>(data, n, hist);""",
                """    RadixHistogram<Passes> hist;
    { const double _t = rt_now();
    scalar_histogram<T, Passes>(data, n, hist);
    RadixTimers::get().hist += rt_now() - _t; }""")

    # -- 2. encode ----------------------------------------------------------
    s = sub_all(s,
                "    for (std::size_t i = 0; i < n; ++i) src[i] = RT::encode(data[i]);",
                """    { const double _t = rt_now();
    for (std::size_t i = 0; i < n; ++i) src[i] = RT::encode(data[i]);
    RadixTimers::get().enc += rt_now() - _t; }""")

    # -- 3. scatter, timed per pass ----------------------------------------
    s = sub_all(s,
                """    for (unsigned s_i = 0; s_i < plan.count; ++s_i) {
        const unsigned p     = plan.active[s_i];""",
                """    for (unsigned s_i = 0; s_i < plan.count; ++s_i) {
        const double _t = rt_now();
        const unsigned p     = plan.active[s_i];""")
    s = sub_all(s,
                """        radix_scatter_pass<Key>(src, n, dst, shift, offset, sc, can_stream);

        Key* t = src; src = dst; dst = t;
    }""",
                """        radix_scatter_pass<Key>(src, n, dst, shift, offset, sc, can_stream);
        RadixTimers::get().scat += rt_now() - _t; RadixTimers::get().passes++;

        Key* t = src; src = dst; dst = t;
    }""")

    # -- 4. decode ----------------------------------------------------------
    s = sub_all(s,
                """    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(src[i]);
    return true;""",
                """    { const double _t = rt_now();
    for (std::size_t i = 0; i < n; ++i) data[i] = RT::decode(src[i]);
    RadixTimers::get().dec += rt_now() - _t; }
    return true;""")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fyx_sort.hpp")
    open(out, "w").write(s)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, os.pardir, os.pardir, "fyx_sort.hpp"))
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fyxinst"
    print(build(src, out))
