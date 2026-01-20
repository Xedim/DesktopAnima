#include "Embedding.h"

namespace Embedding {

    std::vector<double> takens_map(const std::vector<double>& signal, int dim, int tau) {
        std::vector<double> embedded;
        for (size_t i = 0; i + (dim - 1) * tau < signal.size(); ++i) {
            for (int j = 0; j < dim; ++j) {
                embedded.push_back(signal[i + j * tau]);
            }
        }
        return embedded;
    }

}
