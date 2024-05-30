#include <vector>
#include <memory>

struct batch_list {
    std::vector<float> batch = {};
    batch_list * next = NULL;
};