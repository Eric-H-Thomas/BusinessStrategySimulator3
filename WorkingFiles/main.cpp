/*
This main function is used to run the simulator on its own, with control and/or pre-trained AI agents.
Use business_strategy_gym_env.py or another Python script for training AI agents.
*/

#include <exception>
#include <iostream>
#include <string>
#include "Simulator/Simulator.h"
#include "Config/ConfigValidator.h"

using std::cerr;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {

    // Check for correct number of command-line arguments
    if (argc != 2) {
        cerr << "Expected 1 command-line argument. Got " << argc - 1 << endl;
        return 1;
    }

    try {
        Simulator simulator;
        validate_config(argv[1]);
        simulator.load_json_configs(argv[1]);
        simulator.prepare_to_run();

        for (int iSim = 0; iSim < simulator.get_num_sims(); iSim++) {
            cout << "Beginning simulation " << iSim << " of " << simulator.get_num_sims() - 1 << " (indexed at 0)" << endl;
            simulator.reset();

            simulator.run();
        }

        if (simulator.bGenerateMasterOutput) {
            simulator.masterHistory.generate_master_output();
            simulator.masterHistory.generate_market_overlap_file();
        }
    }
    catch (const std::exception& e) {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}