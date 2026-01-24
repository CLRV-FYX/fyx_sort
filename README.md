# FYX-Sort V9：一周重构7个版本的高性能排序库

![C++17](https://img.shields.io/badge/C++-17-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-green)
![License](https://img.shields.io/badge/License-署名--非商业使用-orange)

## 🔥 一句话介绍
如果你的排序库在小数据量时比单线程还慢，那你可能需要这个——一周重构7个版本，从V2到V9，性能翻倍，顺便修复了一堆你想不到的坑。

## 📊 性能对比（10M随机int32）
| 场景 | V2 | V9 | std::sort | 备注 |
|------|----|----|-----------|------|
| 随机分布 | 0.42s | 0.17s | 0.38s | 快了一倍多 |
| 已排序 | 0.008s | 0.001s | 0.005s | 直接检测后返回 |
| 逆序 | 0.15s | 0.002s | 0.14s | 检测后reverse |
| 0-100范围 | 0.22s | 0.09s | 0.18s | 自动用计数排序 |

## 🚀 快速开始
```cpp
#include "fyx_sort_v9.hpp"

int main() {
    std::vector<int> data = {5, 2, 8, 1, 9, 3};
    
    // 基本用法（自动选择最优算法）
    fyx::sort(data);
    
    // 稳定排序
    fyx::stable_sort(data);
    
    // 自定义比较器
    fyx::sort(data, std::greater<int>());
    
    // 带配置选项
    fyx::Options opts;
    opts.parallel = true;
    opts.prefetch_aggressive = true;
    fyx::sort(data, opts);
    
    return 0;
}
```

## 🎯 核心特性

### 1. 无锁工作窃取队列
V2的线程池在1000个元素时比单线程慢3倍——你敢信吗？

V9用了Chase-Lev无锁双端队列：
- 每个线程有自己的任务栈
- 偷任务时才发生原子操作
- 解决了小数据并行效率问题

```cpp
// 以前（V2）的全局锁
std::mutex global_lock; // 所有线程抢这一把锁

// 现在（V9）的无锁队列
class LockFreeWorkStealQueue {
    // 自己从尾部取，偷别人从头偷
    // 只有偷的时候用CAS
};
```

### 2. 分层缓存优化
我发现8K是L1缓存(32KB)的舒适边界，数据大了CPU就开始摆烂。

V9的分层策略：
- **<8K元素**：原地American Flag排序（零额外内存）
- **<100K元素**：带64KB缓冲的LSD基数排序
- **<1M元素**：缓存友好的MSD递归
- **>1M元素**：并行超采样分区

这数据不是猜的——我用了`perf stat -e cache-misses`实测出来的。

### 3. 预计算SIMD掩码表
V8每次排序都要重新计算AVX-512掩码：`0xAA, 0xCC, 0xF0...`

这些明明是常量，为什么每次重算？

V9静态预计算：
```cpp
struct alignas(64) PrecomputedTables {
    __mmask16 blend_AAAA; // 0xAAAA
    __mmask16 blend_CCCC; // 0xCCCC
    // ... 7个常用掩码
    // 对齐到缓存行，防止false sharing
};

// 全局单例，只构造一次
static const PrecomputedTables& get_tables() {
    static const PrecomputedTables tables;
    return tables;
}
```
就这一改动，16元素排序网络快了30%。

### 4. 数据感知选择器
V8对所有数据用同一套算法，但：
- 几乎有序的数据应该用插入排序变体
- 小范围整数应该用计数排序
- 高重复率数据应该用特殊处理

V9先采样256个元素分析特征：
```cpp
DataProfile profile = analyze(data, n);
if (profile.is_sorted) return; // 直接返回
if (profile.is_reverse) {
    std::reverse(data, data + n); // 反转完事
    return;
}
if (profile.is_small_range) {
    counting_sort(data, n); // 计数排序
    return;
}
// ... 其他场景
```

## 🐛 我踩过的坑（你可能也会遇到）

### 坑1：环形数组并发扩容
```cpp
// 错误写法（会segfault）
void push(WorkItem item) {
    if (need_grow()) {
        new_array = new Array(old_size * 2);
        copy(old_array, new_array); // 其他线程还在读old_array!
        delete old_array; // 💥 崩溃
    }
}
```
**解决方案**：版本标记+垃圾回收。旧数组先标记，等确定没人用了再删。

### 坑2：缓存预取距离
我以为预取越远越好：
```cpp
_mm_prefetch(ptr + 1024, _MM_HINT_T0); // 预取太远
```
实际上CPU的预取器比我聪明，只预取下一块数据就行。

### 坑3：编译器玄学
- GCC：把我的常量表优化掉了（用`asm volatile`内存屏障解决）
- Clang：静态初始化顺序随机（用`std::call_once`解决）
- MSVC：AVX-512对齐错误（用`__declspec(align(64))`解决）


## 🛠️ 编译要求
```bash
# 最低要求
g++ -std=c++17 -O3 -march=native your_code.cpp

# 启用AVX-512（如果有）
g++ -std=c++17 -O3 -march=native -mavx512f -mavx512dq your_code.cpp

# 禁用并行（调试用）
g++ -std=c++17 -O3 -DFYX_ENABLE_PARALLEL=0 your_code.cpp
```

## 📈 性能测试
```bash
cd benchmark
./run_benchmarks.sh

# 或者手动测试
g++ -std=c++17 -O3 -march=native bench_std.cpp -o bench_std
./bench_std 1000000  # 测试100万数据
```

测试会输出：
- 每种场景的耗时
- 加速比（对比std::sort）
- 内存使用峰值

## ❓ 常见问题

### Q：为什么我的并行排序没加速？
A：检查两件事：
1. 数据量够大吗？（建议>10万元素）
2. 编译器开了`-O3`和`-march=native`吗？

### Q：支持哪些数据类型？
A：int8/16/32/64, uint8/16/32/64, float, double, 还有自定义大对象（自动用间接排序）。

### Q：线程数怎么控制？
```cpp
fyx::Options opts;
opts.max_threads = 4; // 限制最多4个线程
fyx::sort(data, opts);
```

### Q：稳定排序性能如何？
A：比std::stable_sort快，但比不稳定的慢一些——稳定是有代价的。

## 📄 许可证
**署名-非商业使用许可**

### 你可以：
- 随便用、随便改、随便分发
- 用在开源项目、个人项目、学术研究
- 放在你的博客里当示例代码

### 你必须：
- 保留我的名字（FYX/付炎鑫）
- 不能商用（除非找我授权）

### 商用授权：
如果你真想用这个赚钱，联系我。价格可能是一杯咖啡的钱，也可能是一顿饭——看用途。

## 🤝 贡献
欢迎提Issue和PR，特别是：
1. 新架构支持（ARM SVE2, RISC-V V扩展）
2. 性能优化建议
3. 文档改进（我语文不太好）
4. 新的测试用例

## 🙏 致谢
- Chase-Lev：无锁工作窃取队列论文
- Intel：AVX-512手册（虽然难读）
- Stack Overflow：那些凌晨三点回答我问题的人
- 所有测试用户：你们的bug报告让我代码变好

## 📞 联系
- GitHub Issues: 提问最快
- 邮箱：有问题先搜Issue，搜不到再发
- 博客：我会写技术文章解释实现细节

---

**最后**：如果你也在写性能库，记住——CPU会记仇，缓存会欺骗，编译器会撒谎。但至少，现在我们有个排序库，比std::sort快一倍。

**你遇到过最坑的性能bug是什么？**
