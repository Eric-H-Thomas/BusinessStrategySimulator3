#include <cassert>
#include <stdexcept>
#include <iostream>
#include "../WorkingFiles/DataCache/DataCache.h"

int main() {
    DataCache cache;
    try {
        // Attempt to access a firm-market pair that was never inserted
        cache.mapFirmMarketComboToEntryCost.at({1, 1});
        assert(false && "Expected std::out_of_range when accessing missing firm-market pair");
    } catch (const std::out_of_range&) {
        // Expected path
    }
    std::cout << "Test passed" << std::endl;
    return 0;
}
