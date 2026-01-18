#include "fyx_sort.hpp"
#include <vector>
#include <cassert>

int main() {
    std::vector<int> a = {5, 2, 9, 1, 5, 6};
    fyx::sort(a.data(), a.size());
    assert(a[0] == 1 && a[1] == 2 && a[2] == 5);
    
    // Add more tests as needed
    return 0;
}
