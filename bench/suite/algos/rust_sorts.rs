// Rust standard library sorts: slice::sort_unstable (ipnsort) and slice::sort
// (driftsort).  Built as a staticlib only when rustc is available.
#[repr(C)] pub struct KV { val: u64, key: u64 }
#[repr(C)] pub struct Rec { key: u32, id: u32 }
macro_rules! prim {
    ($u:ident, $s:ident, $t:ty) => {
        #[no_mangle] pub unsafe extern "C" fn $u(p: *mut $t, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_unstable(); } }
        #[no_mangle] pub unsafe extern "C" fn $s(p: *mut $t, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort(); } }
    };
}
prim!(fb_rust_unstable_i32, fb_rust_stable_i32, i32);
prim!(fb_rust_unstable_u32, fb_rust_stable_u32, u32);
prim!(fb_rust_unstable_i64, fb_rust_stable_i64, i64);
prim!(fb_rust_unstable_u64, fb_rust_stable_u64, u64);
macro_rules! float {
    ($u:ident, $s:ident, $t:ty) => {
        #[no_mangle] pub unsafe extern "C" fn $u(p: *mut $t, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); } }
        #[no_mangle] pub unsafe extern "C" fn $s(p: *mut $t, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_by(|a, b| a.partial_cmp(b).unwrap()); } }
    };
}
float!(fb_rust_unstable_f32, fb_rust_stable_f32, f32);
float!(fb_rust_unstable_f64, fb_rust_stable_f64, f64);
#[no_mangle] pub unsafe extern "C" fn fb_rust_unstable_kv(p: *mut KV, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_unstable_by_key(|x| x.key); } }
#[no_mangle] pub unsafe extern "C" fn fb_rust_stable_kv(p: *mut KV, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_by_key(|x| x.key); } }
#[no_mangle] pub unsafe extern "C" fn fb_rust_unstable_rec(p: *mut Rec, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_unstable_by_key(|x| x.key); } }
#[no_mangle] pub unsafe extern "C" fn fb_rust_stable_rec(p: *mut Rec, n: usize) { if n > 1 { std::slice::from_raw_parts_mut(p, n).sort_by_key(|x| x.key); } }
