//
// Created by Eric Thomas on 12/5/23.
//

#pragma once

#include "../Simulator/Simulator.h"
#include <iostream>
#include <string>
#include <vector>
#include <tuple>

using std::cout;
using std::endl;
using std::tuple;
using std::vector;

class PythonAPI {
public:
    void init_simulator(const string& strJsonConfigs);
    vector<double> reset(); // Returns observation
    tuple<vector<double>, double, bool, bool> step(int iActionID); // Returns tuple containing observation, reward, terminated, truncated
    void close();

    // Helper functions for defining the state and action spaces
    int get_num_markets();
    int get_num_agents();

private:
    tuple<vector<double>, double, bool, bool> step_helper();
    Simulator simulator;
};
