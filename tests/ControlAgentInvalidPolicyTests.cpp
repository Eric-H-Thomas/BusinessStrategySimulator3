#include <cassert>
#include <stdexcept>
#include <iostream>
#include "../WorkingFiles/Agent/ControlAgent.h"

void test_invalid_entry_policy() {
    bool thrown = false;
    try {
        ControlAgent agent(1, "invalid", "all", "cournot", 0.1, 0.1, 0.8);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown && "Constructor should throw on invalid entry policy");
}

void test_invalid_exit_policy() {
    bool thrown = false;
    try {
        ControlAgent agent(1, "all", "invalid", "cournot", 0.1, 0.1, 0.8);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown && "Constructor should throw on invalid exit policy");
}

void test_invalid_production_policy() {
    bool thrown = false;
    try {
        ControlAgent agent(1, "all", "all", "invalid", 0.1, 0.1, 0.8);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown && "Constructor should throw on invalid production policy");
}

int main() {
    test_invalid_entry_policy();
    test_invalid_exit_policy();
    test_invalid_production_policy();
    std::cout << "All invalid policy tests passed" << std::endl;
    return 0;
}
