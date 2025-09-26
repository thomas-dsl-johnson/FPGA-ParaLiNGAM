#ifndef PARALINGAM_HPP
#define PARALINGAM_HPP

#include <vector>
#include <cstddef> // for getting size_t

struct Matrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<float> data;
};

class ParaLingamCausalOrderAlgorithm {
public:
    std::vector<int> run(const Matrix& data);
};

#endif
