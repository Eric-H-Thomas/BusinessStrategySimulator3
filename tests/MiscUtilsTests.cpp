#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../WorkingFiles/Utils/MiscUtils.h"

void test_valid_probabilities() {
    std::vector<double> probs{25.0, 25.0, 25.0, 25.0};
    for (int i = 0; i < 10; ++i) {
        int index = MiscUtils::choose_index_given_probabilities(probs);
        assert(index >= 0 && index < static_cast<int>(probs.size()));
    }
}

void test_negative_probability() {
    std::vector<double> probs{50.0, -10.0, 60.0};
    bool thrown = false;
    try {
        MiscUtils::choose_index_given_probabilities(probs);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);
}

int main() {
    test_valid_probabilities();
    test_negative_probability();
    std::cout << "All tests passed" << std::endl;
    return 0;
}
