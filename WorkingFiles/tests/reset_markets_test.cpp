#include "../Simulator/Simulator.h"
#include <iostream>

int main() {
    Simulator simulator;
    if (simulator.load_json_configs("WorkingFiles/Config/economy_only_random.json")) {
        std::cerr << "Failed to load config" << std::endl;
        return 1;
    }
    if (simulator.prepare_to_run()) {
        std::cerr << "prepare_to_run failed" << std::endl;
        return 1;
    }
    if (simulator.get_num_markets() == 0) {
        std::cerr << "No markets after initialization" << std::endl;
        return 1;
    }
    if (simulator.reset()) {
        std::cerr << "reset failed" << std::endl;
        return 1;
    }
    if (simulator.get_num_markets() == 0) {
        std::cerr << "No markets after reset" << std::endl;
        return 1;
    }
    std::cout << "Markets after reset: " << simulator.get_num_markets() << std::endl;
    return 0;
}
