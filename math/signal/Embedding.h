#pragma once
#include <vector>

namespace Embedding {

    // Takens embedding / delay embedding map
    std::vector<double> takens_map(const std::vector<double>& signal, int dim, int tau);

}
