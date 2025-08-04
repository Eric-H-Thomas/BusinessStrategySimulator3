//
// Created by Eric Thomas on 8/17/23.
//

#include "Simulator.h"
#include <map>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <stdexcept>

using std::map;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

/*
Note that the run method should not be used while training AI agents! This method is used for simulations
involving heuristic agents and/or trained AI agents.
*/

int Simulator::run() {
    // Loop through the macro steps
    for (int iMacroStep = 0; iMacroStep < iMacroStepsPerSim; iMacroStep++) {
        if (bVerbose) cout << "Beginning macro step " << iMacroStep + 1 << " of " << iMacroStepsPerSim << endl;

        set_agent_turn_order();

        // Loop through the micro steps
        for (int iAgentID : vecAgentTurnOrder) {
            if (!is_ai_agent(iAgentID)) {
                if (perform_micro_step_control_agent_or_skip_turn(iAgentID))
                    return 1;
            }
            else { // (is AI agent)

                // This is where we will later create a state observation, pass it to the RL algorithm, and get
                // back an action. For now, we'll just set the AI agent's action to 0.
                int action = 0;

                // Execute the micro step with the action chosen by the AI agent
                if (perform_micro_step_ai_agent_turn(iAgentID, action)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}

Simulator::Simulator() = default;


int Simulator::get_num_sims() const { return iNumSims; }
int Simulator::get_macro_steps_per_sim() const { return iMacroStepsPerSim; }
vector<int> Simulator::get_agent_turn_order() { return vecAgentTurnOrder; }


int Simulator::load_json_configs(const string& strConfigFilePath) {

    // Read the JSONReader file
    std::ifstream file(strConfigFilePath);
    if (!file.is_open()) {
        cout << "Error reading json config file\n";
        return 1;
    }

    // Parse JSONReader data
    try {
        file >> simulatorConfigs;
    }
    catch (const nlohmann::json::exception& e) {
        // Handle JSONReader parsing error
        cerr << "JSONReader parsing error: " << e.what() << endl;
        return 1;
    }

    return 0;
}

int Simulator::prepare_to_run() {

    if (set_simulation_parameters())
        return 1;

    if (init_economy())
        return 1;

    if (init_markets())
        return 1;

    if (init_control_agents())
        return 1;

    if (init_AI_agents())
        return 1;

    if (init_firms_for_agents())
        return 1;

//    if (bFixedCostForExistence) {
//        if (set_fixed_cost_for_existence())
//            return 1;
//    }

    init_master_history();

    return 0;
}

int Simulator::reset() {

    // Reset the economy and markets if their corresponding randomization options are set to true
    if (this->bRandomizeEconomyPerSimulation) {
        this->reset_economy();
    }

    if (this->bRandomizeMarketsPerSimulation) {
        this->reset_markets();
    }

    // Set current micro time step and macro time step to 0
    iCurrentMicroTimeStep = 0;
    iCurrentMacroTimeStep = 0;

    // Reset capabilities, portfolio, and capital for all firms
    for (auto pair : mapFirmIDToFirmPtr) {
        auto firm = pair.second;
        const auto& firm_parameters = this->simulatorConfigs["default_firm_parameters"];
        double dbDefaultStartingCapital = firm_parameters["starting_capital"];
        firm->reset((dbDefaultStartingCapital));
    }

    // Initialize the history and the data cache
    init_simulation_history();
    if (init_data_cache(masterHistory.getCurrentSimulationHistoryPtr()))
        return 1;

    // Initialize maps for tracking statistics necessary for AI reward calculations
    for (auto pair : mapAgentIDToAgentPtr) {
        if (is_ai_agent(pair.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(pair.first);
            mapAIAgentIDToCapitalAtLastTurn[pair.first] = firmPtr->getDbCapital();
            mapAIAgentIDToMicroTimeStepOfLastTurn[pair.first] = 0;
        }
    }

    // Call get_market_overlap representation once so that it's initialized in the event that we run the simulator
    // without any AI agents.
    this->get_market_overlap_representation();

    return 0;
}

int Simulator::set_simulation_parameters() {
    try {
        const auto& simulation_parameters = this->simulatorConfigs["simulation_parameters"];
        this->strResultsDir = simulation_parameters["results_dir"];
        this->iNumSims = simulation_parameters["num_sims"];
        this->iMacroStepsPerSim = simulation_parameters["macro_steps_per_sim"];
        this->dbSkippedTurnsPerRegularTurn = simulation_parameters["skipped_turns_per_regular_turn"];
        this->bVerbose = simulation_parameters["verbose"];
        // this->bFixedCostForExistence = simulation_parameters["fixed_cost_for_existence"];
        this->bGenerateMasterOutput = simulation_parameters["generate_master_output"];
        this->bRandomizeTurnOrderWithinEachMacroStep = simulation_parameters["randomize_turn_order_within_each_macro_step"];
        this->bRandomizeAgentFirmAssignmentPerSimulation = simulation_parameters["randomize_agent_firm_assignment_per_simulation"];
        this->bRandomizeVariableCostsPerSimulation = simulation_parameters["randomize_variable_costs_per_simulation"];
        this->bRandomizeEconomyPerSimulation = simulation_parameters["randomize_economy_per_simulation"];
        this->bRandomizeMarketsPerSimulation = simulation_parameters["randomize_markets_per_simulation"];
    }

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error extracting simulation parameters: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int Simulator::init_economy() {

    try {
        const auto& economy_parameters = this->simulatorConfigs["default_economy_parameters"];
        const auto& market_parameters = this->simulatorConfigs["default_market_parameters"];

        vector<int> vecClusterMeans;
        for (const auto& clusterMean : economy_parameters["cluster_means"]) {
            vecClusterMeans.push_back(clusterMean);
        }

        vector<int> vecClusterSDs;
        for (const auto& clusterSD : economy_parameters["cluster_SDs"]) {
            vecClusterSDs.push_back(clusterSD);
        }

        vector<int> vecMarketsPerCluster;
        for (const auto& numMarkets : economy_parameters["markets_per_cluster"]) {
            vecMarketsPerCluster.push_back(numMarkets);
        }

        this->economy = Economy(economy_parameters["possible_capabilities"],
                                economy_parameters["capabilities_per_market"],
                                economy_parameters["num_market_clusters"],
                                vecClusterMeans,
                                vecClusterSDs,
                                vecMarketsPerCluster,
                                market_parameters["market_entry_cost_max"],
                                market_parameters["market_entry_cost_min"]);
    }

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error initializing economy: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


int Simulator::reset_economy() {

    try {
        const auto& economy_parameters = this->simulatorConfigs["default_economy_parameters"];
        const auto& market_parameters = this->simulatorConfigs["default_market_parameters"];

        vector<int> vecClusterMeans;
        for (const auto& clusterMean : economy_parameters["cluster_means"]) {
            vecClusterMeans.push_back(clusterMean);
        }

        vector<int> vecClusterSDs;
        for (const auto& clusterSD : economy_parameters["cluster_SDs"]) {
            vecClusterSDs.push_back(clusterSD);
        }

        vector<int> vecMarketsPerCluster;
        for (const auto& numMarkets : economy_parameters["markets_per_cluster"]) {
            vecMarketsPerCluster.push_back(numMarkets);
        }

        this->economy = Economy(economy_parameters["possible_capabilities"],
                                economy_parameters["capabilities_per_market"],
                                economy_parameters["num_market_clusters"],
                                vecClusterMeans,
                                vecClusterSDs,
                                vecMarketsPerCluster,
                                market_parameters["market_entry_cost_max"],
                                market_parameters["market_entry_cost_min"]);
    }

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error resetting economy: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


int Simulator::init_control_agents() {
    try {
        for (const auto& agentData : this->simulatorConfigs["control_agents"]) {
            auto agentPtr = new ControlAgent(agentData["agent_id"],
                                             agentData["entry_policy"],
                                             agentData["exit_policy"],
                                             agentData["production_policy"],
                                             agentData["entry_action_likelihood"],
                                             agentData["exit_action_likelihood"],
                                             agentData["none_action_likelihood"]);

            this->mapAgentIDToAgentPtr.insert(std::make_pair(agentData["agent_id"], agentPtr));
        }
    }

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error initializing control agents: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int Simulator::init_AI_agents() {
    try {
        for (const auto& agentData : this->simulatorConfigs["ai_agents"]) {
            if (agentData["agent_type"] == "stable_baselines_3") {
                if (agentData["production_policy"] == "Cournot") {
                    auto agentPtr = new StableBaselines3Agent(agentData["agent_id"], ProductionPolicy::Cournot, agentData["path_to_agent"]);
                    this->mapAgentIDToAgentPtr.insert(std::make_pair(agentData["agent_id"], agentPtr));
                    this->iNumAIAgents++;
                }
                else {
                    std::cerr << "AI agent production type not yet supported" << std::endl;
                    throw new std::exception();
                }

            }
        }
    }

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error initializing AI agents: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int Simulator::init_firms_for_agents() {
    // For now, this assigns each agent a firm with
    //    - no capabilities
    //    - no presence in any markets
    //    - the default starting capital

    if (mapAgentIDToAgentPtr.empty()) {
        cerr << "Tried to initialize firms for agents before creating any agents." << endl;
        return 1;
    }

    const auto& firm_parameters = this->simulatorConfigs["default_firm_parameters"];
    double dbDefaultStartingCapital = firm_parameters["starting_capital"];
    const auto& economy_parameters = this->simulatorConfigs["default_economy_parameters"];
    int iPossibleCapabilities = economy_parameters["possible_capabilities"];

    for (auto pair : mapAgentIDToAgentPtr) {
        // Get the ID number of the agent
        int iID = pair.first;

        // Create a firm with the same ID number and place it in the map of firms
        auto firmPtr = new Firm(iID, dbDefaultStartingCapital, iPossibleCapabilities);
        mapFirmIDToFirmPtr.insert(std::make_pair(iID, firmPtr));

        // Set the agent's firm assignment number
        pair.second->iFirmAssignment = iID;
    }

    if (bRandomizeAgentFirmAssignmentPerSimulation) {
        shuffle_agent_firm_assignments();
    }

    return 0;
}

//int Simulator::set_fixed_cost_for_existence() {
//    int iMicroStepsPerSim = this->iMacroStepsPerSim * get_num_total_agents();
//
//    // Set the fixed cost for existence such that capital depletes linearly to zero over the course of the simulation
//    const auto& firm_parameters = this->simulatorConfigs["default_firm_parameters"];
//    double dbDefaultStartingCapital = firm_parameters["starting_capital"];
//    this->dbFixedCostForExistence = dbDefaultStartingCapital / iMicroStepsPerSim;
//
//    return 0;
//}

void Simulator::init_master_history() {
    // Get the number of micro steps per macro step
    double dbMicroStepsPerMacroStep = mapAgentIDToAgentPtr.size() * (1.0 + dbSkippedTurnsPerRegularTurn);
    int iMicroStepsPerMacroStep = static_cast<int>(std::ceil(dbMicroStepsPerMacroStep));

    masterHistory.iMicroStepsPerSim = iMicroStepsPerMacroStep * iMacroStepsPerSim;
    masterHistory.iNumFirms = mapFirmIDToFirmPtr.size();
    masterHistory.iNumMarkets = economy.get_total_markets();

    const auto& economy_parameters = this->simulatorConfigs["default_economy_parameters"];
    masterHistory.iCapabilitiesPerMarket = economy_parameters["capabilities_per_market"];

    masterHistory.strMasterHistoryOutputPath = this->strResultsDir;
}

void Simulator::shuffle_agent_firm_assignments() {
    cerr << "Called shuffle agent-firm assignments option.\nThis has not been implemented yet because the simulator "\
        "currently creates identical firms that begin with the same capital and no capabilities.\nCome back and fill in "\
        "this function or turn the shuffle option off." << endl;
    throw std::exception();
}

int Simulator::init_markets() {
    try {
        const auto& market_parameters = this->simulatorConfigs["default_market_parameters"];
        double dbVarCostMin = market_parameters["variable_cost_min"];
        double dbVarCostMax = market_parameters["variable_cost_max"];
        double dbFixedCostPercentageOfEntry = market_parameters["fixed_cost_percentage_of_entry"];
        double dbExitCostPercentageOfEntry = market_parameters["exit_cost_percentage_of_entry"];
        double dbDemandInterceptMax = market_parameters["demand_intercept_max"];
        double dbDemandInterceptMin = market_parameters["demand_intercept_min"];
        double dbProductDemandSlopeMax = market_parameters["product_demand_slope_max"];
        double dbProductDemandSlopeMin = market_parameters["product_demand_slope_min"];

        // Create a random number generator engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create index for generating market IDs (gets incremented at the end of each iteration)
        int iMarketID = 0;

        for (int iCluster = 0; iCluster < this->economy.get_num_market_clusters(); iCluster++) {
            int iMarketsInCurrCluster = economy.get_vec_markets_per_cluster().at(iCluster);
            for (int j = 0; j < iMarketsInCurrCluster; j++) {

                // Choose the market's demand intercept from a uniform distribution in the range [dbDemandInterceptMin, dbDemandInterceptMax)
                std::uniform_real_distribution<double> demand_intercept_dist(dbDemandInterceptMin, dbDemandInterceptMax);
                double dbDemandIntercept = demand_intercept_dist(gen);

                // Choose the market's demand slope from a uniform distribution in the range [dbProductDemandSlopeMin, dbProductDemandSlopeMax)
                std::uniform_real_distribution<double> demand_slope_dist(dbProductDemandSlopeMin, dbProductDemandSlopeMax);
                double dbProductDemandSlope = demand_slope_dist(gen);

                // Create the capability vector for this market
                double dbMean = this->economy.get_vec_cluster_means().at(iCluster);
                double dbSD = this->economy.get_vec_cluster_SDs().at(iCluster);
                vector<int> vecMarketCapabilities = create_market_capability_vector(dbMean, dbSD);

                // Create a new market and add it to the economy's vector of markets
                this->economy.add_market(Market(iMarketID, dbFixedCostPercentageOfEntry, dbExitCostPercentageOfEntry,
                                                dbDemandIntercept, dbProductDemandSlope, vecMarketCapabilities));

                iMarketID++;
            } // End of inner loop
        } // End of outer loop
    } // End of try block
    catch (const std::exception& e) {
        cerr << "Error initializing markets: " << e.what() << endl;
        return 1;
    }
    return 0;
}


int Simulator::reset_markets() {
    try {
        this->economy.clear_markets();
        const auto& market_parameters = this->simulatorConfigs["default_market_parameters"];
        double dbVarCostMin = market_parameters["variable_cost_min"];
        double dbVarCostMax = market_parameters["variable_cost_max"];
        double dbFixedCostPercentageOfEntry = market_parameters["fixed_cost_percentage_of_entry"];
        double dbExitCostPercentageOfEntry = market_parameters["exit_cost_percentage_of_entry"];
        double dbDemandInterceptMax = market_parameters["demand_intercept_max"];
        double dbDemandInterceptMin = market_parameters["demand_intercept_min"];
        double dbProductDemandSlopeMax = market_parameters["product_demand_slope_max"];
        double dbProductDemandSlopeMin = market_parameters["product_demand_slope_min"];

        // Create a random number generator engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create index for generating market IDs (gets incremented at the end of each iteration)
        int iMarketID = 0;

        for (int iCluster = 0; iCluster < this->economy.get_num_market_clusters(); iCluster++) {
            int iMarketsInCurrCluster = economy.get_vec_markets_per_cluster().at(iCluster);
            for (int j = 0; j < iMarketsInCurrCluster; j++) {

                // Choose the market's demand intercept from a uniform distribution in the range [dbDemandInterceptMin, dbDemandInterceptMax)
                std::uniform_real_distribution<double> demand_intercept_dist(dbDemandInterceptMin, dbDemandInterceptMax);
                double dbDemandIntercept = demand_intercept_dist(gen);

                // Choose the market's demand slope from a uniform distribution in the range [dbProductDemandSlopeMin, dbProductDemandSlopeMax)
                std::uniform_real_distribution<double> demand_slope_dist(dbProductDemandSlopeMin, dbProductDemandSlopeMax);
                double dbProductDemandSlope = demand_slope_dist(gen);

                // Create the capability vector for this market
                double dbMean = this->economy.get_vec_cluster_means().at(iCluster);
                double dbSD = this->economy.get_vec_cluster_SDs().at(iCluster);
                vector<int> vecMarketCapabilities = create_market_capability_vector(dbMean, dbSD);

                // Create a new market and add it to the economy's vector of markets
                this->economy.add_market(Market(iMarketID, dbFixedCostPercentageOfEntry, dbExitCostPercentageOfEntry,
                                                dbDemandIntercept, dbProductDemandSlope, vecMarketCapabilities));

                iMarketID++;
            } // End of inner loop
        } // End of outer loop
    } // End of try block
    catch (const std::exception& e) {
        cerr << "Error initializing markets: " << e.what() << endl;
        return 1;
    }
    return 0;
}





vector<int> Simulator::create_market_capability_vector(const double& dbMean, const double& dbSD) {
    // Create a vector of all zeros
    vector<int> vecCapabilities(this->economy.get_num_possible_capabilities(), 0);

    // Create a normal distribution from the given mean and standard deviation
    std::normal_distribution<double> dist(dbMean, dbSD);

    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // For capabilities_per_market times
    for (int i = 0; i < this->economy.get_num_capabilities_per_market(); i++) {
        // Get a random value from the distribution
        double dbRandomValue = dist(gen);

        // Round the random value to the nearest integer
        int iRoundedValue = std::round(dbRandomValue);

        // Keep track of whether we most recently rounded up or down to the nearest integer
        bool bMostRecentlyRoundedUp = false;
        if (iRoundedValue > dbRandomValue)
            bMostRecentlyRoundedUp = true;

        bool bDone = false;
        int iAttempts = 0;
        while (!bDone) {
            iAttempts++;
            // If the vector is defined at the index of that integer and contains a 0
            if (iRoundedValue >= 0 && iRoundedValue < vecCapabilities.size() && vecCapabilities.at(iRoundedValue) == 0) {
                vecCapabilities.at(iRoundedValue) = 1;
                bDone = true;
            }
            else {
                // Set the integer to its next value
                if (bMostRecentlyRoundedUp) {
                    bMostRecentlyRoundedUp = false;
                    iRoundedValue = iRoundedValue - iAttempts;
                }
                else {
                    bMostRecentlyRoundedUp = true;
                    iRoundedValue = iRoundedValue + iAttempts;
                }
            }
        }
    } // End of for loop
    return vecCapabilities;
}

void Simulator::set_agent_turn_order() {
    if (bVerbose) cout << "Setting the agent turn order" << endl;

    // If we are not resetting the order within each macro step, we only want to set the order once
    if (!bRandomizeTurnOrderWithinEachMacroStep && !vecAgentTurnOrder.empty())
        return;

    // Generate a vector for the new turn order
    int iTotalTurns = get_micro_steps_per_macro_step();
    vector<int> vecNewTurnOrder;

    // Add each agent ID exactly once
    for (const auto& pair : mapAgentIDToAgentPtr) {
        vecNewTurnOrder.push_back(pair.first);
    }

    // Add placeholder IDs for skipped turns
    int iSkipSlots = iTotalTurns - static_cast<int>(mapAgentIDToAgentPtr.size());
    int iPlaceholderID = -1;
    for (int i = 0; i < iSkipSlots; i++) {
        vecNewTurnOrder.push_back(iPlaceholderID--);
    }

    // Shuffle the new turn order vector
    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(vecNewTurnOrder.begin(), vecNewTurnOrder.end(), generator);

    // Set the agent turn order equal to the newly created turn order
    vecAgentTurnOrder = vecNewTurnOrder;
}

int Simulator::perform_micro_step_control_agent_or_skip_turn(const int& iActingAgentID) {
    if (bVerbose) cout << "Performing micro step. Acting agent ID: " << iActingAgentID << endl;

    // Get agent actions
    vector<Action> vecActions;
    try {
        vecActions = get_actions_for_all_agents_control_agent_turn(iActingAgentID);
    }
    catch (std::exception e) {
        cerr << "Error getting agent actions during micro step " << iCurrentMicroTimeStep << endl;
        cerr << e.what() << endl;
        return 1;
    }

    // Execute actions and distribute profits
    if (perform_micro_step_helper(vecActions))
        return 1;

    return 0;
}


int Simulator::perform_micro_step_ai_agent_turn(const int& iActingAgentID, const int& iAIAgentActionID) {
    if (bVerbose) cout << "Performing micro step. Acting agent ID: " << iActingAgentID << endl;

    // Get agent actions
    vector<Action> vecActions;
    try {
        vecActions = get_actions_for_all_agents_ai_agent_turn(iActingAgentID, iAIAgentActionID);
    }
    catch (std::exception e) {
        cerr << "Error getting agent actions during micro step " << iCurrentMicroTimeStep << endl;
        cerr << e.what() << endl;
        return 1;
    }

    // Execute actions and distribute profits
    if (perform_micro_step_helper(vecActions))
        return 1;

    // Increment the number of AI turns that have taken place thus far in the simulation
    iNumAITurns++;

    // Update statistics for reward calculation
    auto firmPtr = get_firm_ptr_from_agent_id(iActingAgentID);
    mapAIAgentIDToCapitalAtLastTurn[iActingAgentID] = firmPtr->getDbCapital();
    mapAIAgentIDToMicroTimeStepOfLastTurn[iActingAgentID] = iCurrentMicroTimeStep;

    return 0;
}

int Simulator::perform_micro_step_helper(vector<Action> vecActions) {
    // Create a map of capital change for each firm within this micro step (capital can be affected by both action
    // execution and profit distribution, so to get the total capital change within the micro time step we must add
    // these two effects). Initialize all values to zero.
    map<int, double> mapFirmIDToCapitalChange;
    for (auto firmID : get_set_firm_IDs()) {
        mapFirmIDToCapitalChange[firmID] = 0.0;
    }

//    // Subtract the fixed existence cost from each firm
//    if (this->bFixedCostForExistence) {
//        for (auto firmID : get_set_firm_IDs()) {
//            // Update capital within the firm object
//            add_profit_to_firm(-dbFixedCostForExistence, firmID);
//            mapFirmIDToCapitalChange.at(firmID) -= this->dbFixedCostForExistence;
//        }
//    }

    // Execute actions and distribute profits
    if (execute_actions(vecActions, &mapFirmIDToCapitalChange))
        return 1;

    if (distribute_profits(&mapFirmIDToCapitalChange))
        return 1;

    // Record capital changes in the history
    for (auto pair : mapFirmIDToCapitalChange) {
        if (pair.second != 0.0) {
            int iFirmID = pair.first;
            auto pFirm = mapFirmIDToFirmPtr[iFirmID];
            double dbCapital = pFirm->getDbCapital();
            currentSimulationHistoryPtr->record_capital_change(iCurrentMicroTimeStep, iFirmID, dbCapital);
        }
    }

    // Declare bankruptcy for any firms with negative capital
    for (auto pair : mapAgentIDToAgentPtr) {
        if (is_bankrupt(pair.first)) {
            auto firm = get_firm_ptr_from_agent_id(pair.first);

            // Record the bankruptcy in the simulation history
            this->currentSimulationHistoryPtr->record_bankruptcy(iCurrentMicroTimeStep, firm->getFirmID(), firm->getSetMarketIDs());

            // Wipe out the firm's portfolio and capabilities; Set capital to -1e-9
            firm->declare_bankruptcy();

            // Record bankruptcy in the data cache
            for (auto iMarketID : get_set_market_IDs()) {
                auto pairFirmMarket = std::make_pair(firm->getFirmID(), iMarketID);

                dataCache.mapFirmMarketComboToRevenue[pairFirmMarket] = 0.0;
                dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket] = 0.0;
                dataCache.mapFirmMarketComboToVarCost[pairFirmMarket] = 0.0;
                dataCache.mapFirmMarketComboToEntryCost[pairFirmMarket] = 0.0;
                dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket] = 0.0;
                dataCache.mapFirmMarketComboToPrice[pairFirmMarket] = 0.0; // Will need to be made dynamic if the simulator is ever allowed to have more than one price per market
            }
        }
    }

    // Increment the micro step
    iCurrentMicroTimeStep++;

    // Increment the macro step if necessary
    if (at_beginning_of_macro_step())
        iCurrentMacroTimeStep++;

    return 0;
}

vector<Action> Simulator::get_actions_for_all_agents_control_agent_turn(const int& iActingAgentID) {
    if (bVerbose) cout << "Getting actions for all agents" << endl;
    vector<Action> vecActions;
    for (const auto& pair : mapAgentIDToAgentPtr) {
        auto agentPtr = pair.second;
        if (agentPtr->enumAgentType == AgentType::Control) {
            ControlAgent* controlAgentPtr = dynamic_cast<ControlAgent*>(agentPtr);
            // Get action for the acting agent
            if (agentPtr->get_agent_ID() == iActingAgentID) {
                try {
                    vecActions.emplace_back(get_agent_action(*controlAgentPtr));
                }
                catch (std::exception e) {
                    cerr << "Error getting actions for control agents" << e.what() << endl;
                    throw std::exception();
                }
            }
            else { // Create none actions for the agents not currently acting
                vecActions.emplace_back(Action::generate_none_action(controlAgentPtr->get_agent_ID()));
            }
        }
            // Create none actions for the AI agents
        else if (agentPtr->enumAgentType == AgentType::StableBaselines3) {
            vecActions.emplace_back(Action::generate_none_action(agentPtr->get_agent_ID()));
        }
    }
    return vecActions;
}

vector<Action> Simulator::get_actions_for_all_agents_ai_agent_turn(const int& iActingAgentID, const int& iAIAgentActionID) {
    if (bVerbose) cout << "Getting actions for all agents" << endl;
    vector<Action> vecActions;
    for (const auto& pair : mapAgentIDToAgentPtr) {
        auto agentPtr = pair.second;
        if (agentPtr->enumAgentType == AgentType::Control) {
            // Create none actions for the control agents
            vecActions.emplace_back(Action::generate_none_action(pair.first));
        }
        else if (agentPtr->enumAgentType == AgentType::StableBaselines3) {
            if (agentPtr->get_agent_ID() == iActingAgentID) {
                // If the AI agent is bankrupt, assign the none action.
                if (is_bankrupt(iActingAgentID)) {
                    vecActions.emplace_back(Action::generate_none_action(iActingAgentID));
                }
                else {
                    vecActions.emplace_back(convert_action_ID_to_action_object(iActingAgentID, iAIAgentActionID));
                }
            }
            else {
                vecActions.emplace_back(Action::generate_none_action(agentPtr->get_agent_ID()));
            }
        }
    }
    return vecActions;
}

Action Simulator::convert_action_ID_to_action_object(const int& iActingAgentID, const int& iAIAgentActionID) {
    // The action space for the StableBaselines3 interface is a vector of length num_markets+1. An action ID less
    // than or equal to num_markets represents a reversal of market presence in the given market. An action ID equal
    // to num_markets represents the do-nothing action.

    // Do nothing action
    if (iAIAgentActionID == economy.get_total_markets()) {
        return Action::generate_none_action(iActingAgentID);
    }

    auto firmPtr = get_firm_ptr_from_agent_id(iActingAgentID);
    if (firmPtr->is_in_market(economy.get_market_by_ID(iAIAgentActionID))) {
        // Firm is present in the given market and needs to be removed
        return Action(iActingAgentID, ActionType::enumExitAction, iAIAgentActionID, iCurrentMicroTimeStep);
    }
    else {
        // Firm is not present in the given market and needs to be added
        return Action(iActingAgentID, ActionType::enumEntryAction, iAIAgentActionID, iCurrentMicroTimeStep);
    }
}

int Simulator::execute_actions(const vector<Action>& vecActions, map<int, double>* pMapFirmIDToCapitalChange) {
    if (bVerbose) cout << "Executing agent actions" << endl;

    for (const Action& action : vecActions) {
        switch (action.enumActionType) {
            case ActionType::enumEntryAction:
                if (execute_entry_action(action, pMapFirmIDToCapitalChange))
                    return 1;
                break;
            case ActionType::enumExitAction:
                if (execute_exit_action(action, pMapFirmIDToCapitalChange))
                    return 1;
                break;
            case ActionType::enumNoneAction:
                // Do nothing for ActionType::enumNoneAction
                break;
            default:
                // Should never reach this section of the code
                std::cerr << "Invalid enumerated action type" << std::endl;
                return 1;
        } // End of switch block
    } // End of for loop

    return 0;
}

int Simulator::execute_entry_action(const Action& action, map<int, double>* pMapFirmIDToCapitalChange) {
    // Get the entry cost for the firm
    auto firmPtr = get_firm_ptr_from_agent_id(action.iAgentID);
    auto pairFirmMarket = std::make_pair(firmPtr->getFirmID(), action.iMarketID);
    double dbEntryCost = dataCache.mapFirmMarketComboToEntryCost.at(pairFirmMarket);

    // Update capital within the firm object
    firmPtr->add_capital(-dbEntryCost);

    // Update the map of firm IDs to capital change with the entry cost of this action
    pMapFirmIDToCapitalChange->at(firmPtr->getFirmID()) -= dbEntryCost;

    // Update the fixed cost for this firm-market combo
    auto marketCopy = economy.get_market_by_ID(action.iMarketID);
    double dbFixedCostAsPctOfEntryCost = marketCopy.getFixedCostAsPercentageOfEntryCost();
    dbFixedCostAsPctOfEntryCost *= 0.01; // Scaling factor for percentage expressed as whole number
    double dbFixedCost = dbFixedCostAsPctOfEntryCost * dbEntryCost;
    dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket] = dbFixedCost;
    currentSimulationHistoryPtr->record_fixed_cost_change(iCurrentMicroTimeStep,
                                                          dbFixedCost, pairFirmMarket.first, pairFirmMarket.second);

    // Update the firm's capability vector
    firmPtr->add_market_capabilities_to_firm_capabilities(economy.get_market_by_ID(action.iMarketID));

    // Add this market to the firm's portfolio and record this change in the history
    if (firmPtr->add_market_to_portfolio(action.iMarketID))
        return 1;
    currentSimulationHistoryPtr->record_market_presence_change(action.iMicroTimeStep, true, firmPtr->getFirmID(), action.iMarketID);

    // Update the entry cost for this firm for all markets
    for (Market market : economy.get_vec_markets()) {
        // Get vector of capabilities the firm lacks to enter the market
        auto vecFirmCapabilities = firmPtr->getVecCapabilities();
        auto vecMarketCapabilities = market.get_vec_capabilities();

        if (vecFirmCapabilities.size() != vecMarketCapabilities.size()) {
            cerr << "Mismatch between firm capability vector size and market capability vector size in execute_entry_action()" << endl;
            return 1;
        }

        std::vector<int> vecMissingCapabilities;
        // Reserve space for the missing capabilities vector
        vecMissingCapabilities.reserve(vecFirmCapabilities.size());

        // Set the vecMissingCapabilities vector to 1 where the market requires a capability the firm does not have
        for (size_t i = 0; i < vecFirmCapabilities.size(); i++) {
            if (vecMarketCapabilities[i] && !vecFirmCapabilities[i]) {
                vecMissingCapabilities.push_back(1);
            }
            else {
                vecMissingCapabilities.push_back(0);
            }
        }

        // Calculate the cost of the missing capabilities vector
        double dbCost = MiscUtils::dot_product(vecMissingCapabilities, economy.get_vec_capability_costs());

        // Update the data cache and the history if the entry cost has changed since it was last calculated
        auto pair = std::make_pair(firmPtr->getFirmID(), market.get_market_id());
        double dbPriorCost = dataCache.mapFirmMarketComboToEntryCost[pair];
        if (dbCost != dbPriorCost) {
            dataCache.mapFirmMarketComboToEntryCost[pair] = dbCost;
            currentSimulationHistoryPtr->record_entry_cost_change(iCurrentMicroTimeStep,
                                                                  dbCost, firmPtr->getFirmID(), market.get_market_id());
        }
    }

    return 0;
}

int Simulator::execute_exit_action(const Action& action, map<int, double>* pMapFirmIDToCapitalChange) {
    // Get a copy of the market
    auto marketCopy = economy.get_market_by_ID(action.iMarketID);

    // Get the exit cost for the firm
    auto firmPtr = get_firm_ptr_from_agent_id(action.iAgentID);
    auto pairFirmMarket = std::make_pair(firmPtr->getFirmID(), action.iMarketID);
    double dbEntryCost = dataCache.mapFirmMarketComboToEntryCost.at(pairFirmMarket);
    double dbExitCost = marketCopy.getExitCostAsPercentageOfEntryCost() * dbEntryCost * 0.01; // Scaling factor due to whole percentages

    // Update capital within the firm object
    firmPtr->add_capital(-dbExitCost);

    // Update the map of firm IDs to capital change with the exit cost of this action
    pMapFirmIDToCapitalChange->at(firmPtr->getFirmID()) -= dbExitCost;

    // Update the fixed cost for this firm-market combo
    dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket] = 0.0;
    currentSimulationHistoryPtr->record_fixed_cost_change(iCurrentMicroTimeStep,
                                                          0.0, pairFirmMarket.first, pairFirmMarket.second);

    // Update the firm's capability vector
    firmPtr->remove_market_capabilities_from_firm_capabilities(marketCopy, economy);

    // Remove this market from the firm's portfolio and record this change in the history
    if (firmPtr->remove_market_from_portfolio(action.iMarketID))
        return 1;
    currentSimulationHistoryPtr->record_market_presence_change(action.iMicroTimeStep, false, firmPtr->getFirmID(), action.iMarketID);

    // Update the entry cost for this firm for each market
    for (Market market : economy.get_vec_markets()) {
        // Get vector of capabilities the firm lacks to enter the market
        auto vecFirmCapabilities = firmPtr->getVecCapabilities();
        auto vecMarketCapabilities = market.get_vec_capabilities();

        if (vecFirmCapabilities.size() != vecMarketCapabilities.size()) {
            cerr << "Mismatch between firm capability vector size and market capability vector size in execute_entry_action()" << endl;
            return 1;
        }

        std::vector<int> vecMissingCapabilities;
        // Reserve space for the missing capabilities vector
        vecMissingCapabilities.reserve(vecFirmCapabilities.size());

        // Set the vecMissingCapabilities vector to 1 where the market requires a capability the firm does not have
        for (size_t i = 0; i < vecFirmCapabilities.size(); i++) {
            if (vecMarketCapabilities[i] && !vecFirmCapabilities[i]) {
                vecMissingCapabilities.push_back(1);
            }
            else {
                vecMissingCapabilities.push_back(0);
            }
        }

        // Calculate the cost of the missing capabilities vector
        double dbCost = MiscUtils::dot_product(vecMissingCapabilities, economy.get_vec_capability_costs());

        // Update the data cache and the history if the entry cost has changed since it was last calculated
        auto pair = std::make_pair(firmPtr->getFirmID(), market.get_market_id());
        double dbPriorCost = dataCache.mapFirmMarketComboToEntryCost[pair];
        if (dbCost != dbPriorCost) {
            dataCache.mapFirmMarketComboToEntryCost[pair] = dbCost;
            currentSimulationHistoryPtr->record_entry_cost_change(iCurrentMicroTimeStep,
                                                                  dbCost, firmPtr->getFirmID(), market.get_market_id());
        }
    }

    // Record changes to revenue, price, and quantity
    currentSimulationHistoryPtr->record_revenue_change(iCurrentMicroTimeStep, 0.0, firmPtr->getFirmID(), action.iMarketID);
    currentSimulationHistoryPtr->record_price_change(iCurrentMicroTimeStep, 0.0, firmPtr->getFirmID(), action.iMarketID);
    currentSimulationHistoryPtr->record_production_quantity_change(iCurrentMicroTimeStep, 0.0, firmPtr->getFirmID(), action.iMarketID);
    dataCache.mapFirmMarketComboToRevenue[pairFirmMarket] = 0.0;
    dataCache.mapFirmMarketComboToPrice[pairFirmMarket] = 0.0;
    dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket] = 0.0;

    return 0;
}

Action Simulator::get_agent_action(const ControlAgent& agent) {
    // If the agent is bankrupt, they automatically choose the none action.
    if (is_bankrupt(agent.get_agent_ID())) {
        return Action::generate_none_action(agent.get_agent_ID());
    }

    ActionType actionType = get_action_type(agent);

    if (actionType == ActionType::enumEntryAction) {
        return get_entry_action(agent);
    }
    else if (actionType == ActionType::enumExitAction) {
        return get_exit_action(agent);
    }
    else if (actionType == ActionType::enumNoneAction) {
        return Action::generate_none_action(agent.get_agent_ID());
    }

    // Should never reach this part of the code
    cerr << "Error getting control agent action" << endl;
    throw std::exception();
}

ActionType Simulator::get_action_type(const ControlAgent& agent) {
    int iActionTypeIndex = MiscUtils::choose_index_given_probabilities(agent.get_action_likelihood_vector());
    if (iActionTypeIndex == 0) {
        return ActionType::enumEntryAction;
    }
    if (iActionTypeIndex == 1) {
        return ActionType::enumExitAction;
    }
    if (iActionTypeIndex == 2) {
        return ActionType::enumNoneAction;
    }

    // Should never reach this part of the code
    cerr << "Error getting action_type" << endl;
    throw std::exception();
}

Action Simulator::get_entry_action(const ControlAgent& agent) {
    Firm* firmPtr = get_firm_ptr_from_agent(agent);
    set<Market> setPossibleMarketsToEnter;
    Market finalChoiceMarket;

    // Iterate through all markets, adding each one to the decision set if the firm is not already in that market
    for (const auto& market : economy.get_vec_markets()) {
        if (!firmPtr->is_in_market(market))
            setPossibleMarketsToEnter.insert(market);
    }

    // Check for the case that there are no markets to enter (extremely rare)
    if (setPossibleMarketsToEnter.empty()) {
        return Action::generate_none_action(agent.get_agent_ID());
    }

    // Choose a market to enter based on the entry policy
    if (agent.get_enum_entry_policy() == EntryPolicy::All) {
        finalChoiceMarket = MiscUtils::choose_random_from_set(setPossibleMarketsToEnter);
    }
    else if (agent.get_enum_entry_policy() == EntryPolicy::HighestOverlap) {
        finalChoiceMarket = firmPtr->choose_market_with_highest_overlap(setPossibleMarketsToEnter);
    }
    else {
        // Should never reach this part of the code
        cerr << "Error getting action_type" << endl;
        throw std::exception();
    }

    // Construct and return the action object
    return Action(agent.get_agent_ID(), ActionType::enumEntryAction, finalChoiceMarket.get_market_id(), iCurrentMicroTimeStep);
}

Action Simulator::get_exit_action(const ControlAgent& agent) {
    Firm* firmPtr = get_firm_ptr_from_agent(agent);
    int iFinalChoiceMarketID;

    // Check for the case that there are no markets to exit
    if (firmPtr->getSetMarketIDs().empty()) {
        return Action::generate_none_action(agent.get_agent_ID());
    }

    // If exit policy is ALL, randomly choose a market to exit
    if (agent.get_enum_exit_policy() == ExitPolicy::All) {
        iFinalChoiceMarketID = MiscUtils::choose_random_from_set(firmPtr->getSetMarketIDs());
    }

    // If exit policy is LOSS, choose the market with the lowest profit in the most recent time step
    else if (agent.get_enum_exit_policy() == ExitPolicy::Loss) {
        // Find the market with the lowest profit from the most recent time step

        double dbLowestProfit = std::numeric_limits<double>::infinity();

        int iFirmID = this->get_firm_ptr_from_agent(agent)->getFirmID();
        for (int iMarketID : this->get_set_market_IDs()) {
            auto pairFirmMarket = std::make_pair(iFirmID, iMarketID);
            double dbRev = dataCache.mapFirmMarketComboToRevenue[pairFirmMarket];
            double dbFixedCost = dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket];
            double dbVarCost = dataCache.mapFirmMarketComboToVarCost[pairFirmMarket];
            double dbQty = dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket];
            double dbTotalCost = dbFixedCost + (dbVarCost * dbQty);
            double dbProfit = dbRev - dbTotalCost;

            if (dbProfit < dbLowestProfit) {
                dbLowestProfit = dbProfit;
                iFinalChoiceMarketID = iMarketID;
            }
        }
    }

    else {
        // Should never reach this part of the code
        cerr << "Error getting exit action" << endl;
        throw std::exception();
    }

    // Construct and return the action object
    return Action(agent.get_agent_ID(), ActionType::enumExitAction, iFinalChoiceMarketID, iCurrentMicroTimeStep);
}

void Simulator::init_simulation_history() {
    if (bVerbose) cout << "Initializing simulation history" << endl;

    // Generate map of agents to firms
    map<int, int> mapAgentToFirm;
    for (const auto& pair : this->mapAgentIDToAgentPtr) {
        auto agentPtr = pair.second;
        mapAgentToFirm[agentPtr->get_agent_ID()] = agentPtr->iFirmAssignment;
    }

    // Generate map of firms' starting capital amounts
    map<int, double> mapFirmStartingCapital;
    for (const auto& pair : this->mapFirmIDToFirmPtr) {
        auto firmPtr = pair.second;
        mapFirmStartingCapital[firmPtr->getFirmID()] = firmPtr->getDbCapital();
    }

    // Generate map of market maximum entry costs
    map<int, double> mapMarketMaximumEntryCosts;
    for (auto market : this->economy.get_vec_markets()) {
        vector<int> vecMarketCapabilities = market.get_vec_capabilities();
        double dbMaxEntryCost = MiscUtils::dot_product(vecMarketCapabilities, this->economy.get_vec_capability_costs());
        mapMarketMaximumEntryCosts[market.get_market_id()] = dbMaxEntryCost;
    }

    // Generate map of firm to agent descriptions
    map<int, string> mapFirmIDToAgentDescriptions;
    for (auto pair : mapAgentToFirm) {
        int iFirmID = pair.second;
        auto pAgent = get_agent_ptr_from_firm_ID(iFirmID);
        mapFirmIDToAgentDescriptions[iFirmID] = pAgent->to_string();
    }

    // Initialize the simulation history using the above four maps
    currentSimulationHistoryPtr = new SimulationHistory(mapAgentToFirm, mapFirmIDToAgentDescriptions,
                                                        mapFirmStartingCapital, mapMarketMaximumEntryCosts);

    // Clear out the master history if in training mode
    if (bTrainingMode) {
        for (auto pSimHistory : masterHistory.vecSimulationHistoryPtrs) {
            delete pSimHistory;
        }
        masterHistory.vecSimulationHistoryPtrs.clear();
    }

    // Add the current simulation history to the master history
    masterHistory.vecSimulationHistoryPtrs.push_back(currentSimulationHistoryPtr);
}

int Simulator::init_data_cache(SimulationHistory* pCurrentSimulationHistory) {
    if (bVerbose) cout << "Initializing data cache" << endl;

    try {
        // Get all firm-market combinations
        set<pair<int, int>> setFirmMarketCombinations;
        for (int iFirmID : get_set_firm_IDs()) {
            for (int iMarketID : get_set_market_IDs()) {
                setFirmMarketCombinations.insert(std::make_pair(iFirmID, iMarketID));
            }
        }

        // Create a uniform distribution for drawing variable costs in the range [dbVarCostMin, dbVarCostMax)
        const auto& default_market_parameters = this->simulatorConfigs["default_market_parameters"];
        double dbVarCostMin = default_market_parameters["variable_cost_min"];
        double dbVarCostMax = default_market_parameters["variable_cost_max"];
        std::uniform_real_distribution<double> var_cost_dist(dbVarCostMin, dbVarCostMax);

        // Create a random number generator engine
        std::random_device rd;
        std::mt19937 gen(rd());

        // Initialize revenues, fixed costs, quantities produced to zero for each firm-market combination.
        // Initialize variable costs using the uniform distribution created above.
        for (auto pairFirmMarket : setFirmMarketCombinations) {
            dataCache.mapFirmMarketComboToRevenue[pairFirmMarket] = 0.0;
            dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket] = 0.0;
            dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket] = 0.0;
            dataCache.mapFirmMarketComboToPrice[pairFirmMarket] = 0.0;

            // (Reuse the variable costs from the previous simulation if this condition is false)
            if (!dataCache.bInitialized || bRandomizeVariableCostsPerSimulation) {
                dataCache.mapFirmMarketComboToVarCost[pairFirmMarket] = var_cost_dist(gen);
            }
        }

        // Copy over the map of variable costs to the simulation history
        currentSimulationHistoryPtr->mapFirmMarketComboToVarCost = dataCache.mapFirmMarketComboToVarCost;

        // Initialize each market-firm entry cost to the maximum entry cost for that market
        for (auto combination : setFirmMarketCombinations) {
            int iMarketID = combination.second;
            double dbEntryCost = pCurrentSimulationHistory->mapMarketMaximumEntryCost[iMarketID];
            dataCache.mapFirmMarketComboToEntryCost[combination] = dbEntryCost;
        }

        // Mark the data cache as having been initialized
        dataCache.bInitialized = true;

    } // End of try block

    catch (const nlohmann::json::exception& e) {
        std::cerr << "Error initializing the data cache: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

Firm* Simulator::get_firm_ptr_from_agent_ptr(BaseAgent* agentPtr) {
    int iFirmID = agentPtr->iFirmAssignment;
    return mapFirmIDToFirmPtr.at(iFirmID);
}

Firm* Simulator::get_firm_ptr_from_agent_id(const int& iAgentID) {
    auto agentPtr = mapAgentIDToAgentPtr[iAgentID];
    int iFirmID = agentPtr->iFirmAssignment;
    return mapFirmIDToFirmPtr.at(iFirmID);
}

BaseAgent* Simulator::get_agent_ptr_from_firm_ID(int iFirmID) {
    for (auto pair : mapAgentIDToAgentPtr) {
        auto pAgent = pair.second;
        if (pAgent->iFirmAssignment == iFirmID) {
            return pAgent;
        }
    }
    cerr << "Error in get_agent_ptr_from_firm_ID" << endl;
    throw std::exception();
}

Firm* Simulator::get_firm_ptr_from_agent(const ControlAgent& agent) {
    int iFirmID = agent.iFirmAssignment;
    return mapFirmIDToFirmPtr.at(iFirmID);
}

int Simulator::distribute_profits(map<int, double>* pMapFirmIDToCapitalChange) {
    if (bVerbose) cout << "Distributing profits" << endl;

    // Iterate through each of the markets in the economy
    for (auto market : economy.get_vec_markets()) {
        /*
            Here, we use the same variable names provided in the simulator documentation:
              Key:
              Q: total production
              P: price level (for now, we assume one price level for all firms)
              q: production for a specific firm-market combination
              n: number of firms in the market
              a: demand curve intercept
              b: demand curve slope
              V: average variable cost in the market
              v: variable cost for a specific firm-market combination
        */

        // Calculate the total production level and price for the market
        set<int> setFirmIDsInMarket = get_firm_IDs_in_market(market);
        double n = (double)setFirmIDsInMarket.size(); // Cast to double to avoid integer division problems
        double V = get_average_var_cost_in_market(market);
        double a = market.getDbDemandIntercept();
        double b = market.getDbDemandSlope();
        double Q = (n / (n + 1.0)) * ((a - V) / b);
        double P = a - (b * Q);

        auto mapFirmIDToVarCosts = get_map_firm_to_var_cost_for_market(market);

        // Iterate through the firms in the current market
        for (int iFirmID : get_firm_IDs_in_market(market)) {
            // Calculate production quantity for the firm-market combo
            double v = mapFirmIDToVarCosts[iFirmID];

            ProductionPolicy policy;
            try {
                auto pAgent = get_agent_ptr_from_firm_ID(iFirmID);
                policy = pAgent->get_enum_production_policy();
            }
            catch (std::exception e) {
                cerr << "Error in distribute_profits method: " << e.what() << endl;
                return 1;
            }

            double q; // Production quantity for this firm-market combo
            if (policy == ProductionPolicy::Cournot) {
                // Adjusting this line to truncate production quantities at zero when necessary. Later, we'll adjust the code to reconfigure production quantities in these edge cases.
                // q = (a - (b * Q) - v) / b;
                q = std::max(0.0, (a - (b * Q) - v) / b);
            }
            else {
                cerr << "Did not specify a valid production policy" << endl;
                return 1;
            }

            // Calculate revenue and profit for the firm-market combo
            auto pairFirmMarket = std::make_pair(iFirmID, market.get_market_id());
            double dbRevenue = q * P;
            double dbFixedCost = dataCache.mapFirmMarketComboToFixedCost[pairFirmMarket];
            double dbVarCost = q * v;
            double dbProfit = dbRevenue - dbFixedCost - dbVarCost;

            // Update capital within the firm object
            add_profit_to_firm(dbProfit, iFirmID);

            // Track accumulated profit for this firm for this time step
            pMapFirmIDToCapitalChange->at(iFirmID) += dbProfit;

            // Update revenue in the history and data cache if needed
            if (dbRevenue != dataCache.mapFirmMarketComboToRevenue[pairFirmMarket]) {
                currentSimulationHistoryPtr->record_revenue_change(iCurrentMicroTimeStep, dbRevenue,
                                                                   pairFirmMarket.first, pairFirmMarket.second);
                dataCache.mapFirmMarketComboToRevenue[pairFirmMarket] = dbRevenue;
            }

            // Update production quantities in the history and data cache if needed
            if (q != dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket]) {
                currentSimulationHistoryPtr->record_production_quantity_change(iCurrentMicroTimeStep, q,
                                                                               pairFirmMarket.first, pairFirmMarket.second);
                dataCache.mapFirmMarketComboToQtyProduced[pairFirmMarket] = q;
            }

            // Update prices in the history and data cache if needed
            // Note: Yes, there is only one price per market for now, but we record price changes according to firm-
            // market combinations in case we want to change the simulator to allow for intra-market price variation.
            if (P != dataCache.mapFirmMarketComboToPrice[pairFirmMarket]) {
                currentSimulationHistoryPtr->record_price_change(iCurrentMicroTimeStep, P,
                                                                 pairFirmMarket.first, pairFirmMarket.second);
                dataCache.mapFirmMarketComboToPrice[pairFirmMarket] = P;
            }
        } // End of loop through firms
    } // End of loop through markets
    return 0;
} // End of distribute_profits method

set<int> Simulator::get_set_firm_IDs() {
    set<int> setFirmIDs;
    for (auto pair : mapFirmIDToFirmPtr) {
        setFirmIDs.insert(pair.first);
    }
    return setFirmIDs;
}

set<int> Simulator::get_set_market_IDs() {
    return economy.get_set_market_IDs();
}

int Simulator::get_num_markets() {
    return economy.get_total_markets();
}

set<int> Simulator::get_firm_IDs_in_market(Market market) {
    set<int> setFirmIDs;
    for (auto pair : mapFirmIDToFirmPtr) {
        auto pFirm = pair.second;
        if (pFirm->is_in_market(market)) {
            setFirmIDs.insert(pFirm->getFirmID());
        }
    }
    return setFirmIDs;
}

map<int, double> Simulator::get_map_firm_to_var_cost_for_market(Market market) {
    map<int, double> mapFirmToVarCost;
    int iMarketID = market.get_market_id();

    for (auto pair : dataCache.mapFirmMarketComboToVarCost) {
        auto currentFirmMarketPair = pair.first;
        double dbCurrentVarCost = pair.second;

        int iCurrentFirmID = currentFirmMarketPair.first;
        int iCurrentMarketID = currentFirmMarketPair.second;

        if (iMarketID == iCurrentMarketID) {
            mapFirmToVarCost.insert(std::make_pair(iCurrentFirmID, dbCurrentVarCost));
        }
    }

    return mapFirmToVarCost;
}

double Simulator::get_average_var_cost_in_market(Market market) {
    map<int, double> mapFirmToVarCost = get_map_firm_to_var_cost_for_market(market);

    set<int> setFirmsInMarket = get_firm_IDs_in_market(market);

    if (setFirmsInMarket.empty()) {
        return 0.0;
    }

    double dbTotalVarCost = 0.0;
    for (int iFirmID : setFirmsInMarket) {
        dbTotalVarCost += mapFirmToVarCost[iFirmID];
    }

    int iTotalFirms = setFirmsInMarket.size();
    return dbTotalVarCost / iTotalFirms;
}

void Simulator::add_profit_to_firm(double dbProfit, int iFirmID) {
    auto pFirm = mapFirmIDToFirmPtr[iFirmID];
    pFirm->add_capital(dbProfit);
}

int Simulator::get_micro_steps_per_macro_step() {
    double dbTotalTurns = mapAgentIDToAgentPtr.size() * (1.0 + dbSkippedTurnsPerRegularTurn);
    return static_cast<int>(std::ceil(dbTotalTurns));
}

bool Simulator::is_ai_agent(const int& iAgentID) {
    // Case where the agent ID does not exist and represents a skip turn
    if (mapAgentIDToAgentPtr.find(iAgentID) == mapAgentIDToAgentPtr.end())
        return false;

    auto agentPtr = mapAgentIDToAgentPtr[iAgentID];

    if (agentPtr->enumAgentType == AgentType::Control) {
        return false;
    }
    else if (agentPtr->enumAgentType == AgentType::StableBaselines3) {
        return true;
    }
    else {
        cerr << "Tried to determine whether agent was an AI agent but agent type has not been configured in"
                "Simulator::is_ai_agent. Add code to account for this agent type." << endl;
        throw std::exception();
    }
}

bool Simulator::is_bankrupt(const int& iAgentID) {
    auto firmPtr = get_firm_ptr_from_agent_id(iAgentID);
    return firmPtr->getDbCapital() < 0.0;
}

vector<double> Simulator::generate_state_observation(const int& iAgentID) {
    /*
     * Let F be the number of firms in the simulation.
     * Let M be the number of markets in the simulation.
     * State observations are then structured as follows:
     *
     * 1. Capital of all firms (vector of dimension F)
     * 2. Market overlap structure (matrix of dimension MxM)
     * 3. Variable costs for all firm-market combinations realized thus far in the simulation
     *    (i.e., if firm i is present or has been present in market j, then we give the AI agents visibility
     *    to the variable cost for firm i--market j) (matrix of dimension FxM)
     * 4. Fixed cost for each firm-market combination (matrix of dimension FxM)
     * 5. Market portfolio of all firms (matrix of dimension FxM)
     * 6. Entry cost for every firm-market combination (matrix of dimension FxM)
     * 7. Demand intercept in each market (vector of dimension M)
     * 8. Slope in each market (vector of dimension M)
     * 9. Most recent quantity for each firm-market combination (matrix of dimension FxM)
     * 10. Most recent price for each firm-market combination (matrix of dimension FxM)
     *
     * These 10 components of the state representation are each flattened into one-dimensional vectors
     * and then concatenated to create a single state observation vector.
     *
     * For components involving information specific to each firm, the acting AI agent's information is given first,
     * the remaining AI agents' information (if there are other AI agents) is given second (in ascending order by
     * agent ID), and the control agents' information is given last (in ascending order by agent ID).
     *
     * For components involving information specific to each market, the info is given in ascending order by market ID.
     *
     * For components involving information specific to firm-market combinations, the above two rules apply. Info
     * is ordered first at the firm level and then at the market level (i.e., info pertaining to a firm for
     * all markets is given before the info for the next firm is given).
     */

    vector<vector<double>> state_observation;
    state_observation.push_back(get_capital_representation(iAgentID));
    state_observation.push_back(get_market_overlap_representation());
    state_observation.push_back(get_variable_cost_representation(iAgentID));
    state_observation.push_back(get_fixed_cost_representation(iAgentID));
    state_observation.push_back(get_market_portfolio_representation(iAgentID));
    state_observation.push_back(get_entry_cost_representation(iAgentID));
    state_observation.push_back(get_demand_intercept_representation());
    state_observation.push_back(get_demand_slope_representation());
    state_observation.push_back(get_quantity_representation(iAgentID));
    state_observation.push_back(get_price_representation(iAgentID));
    return MiscUtils::flatten(state_observation);
}

vector<double> Simulator::get_capital_representation(const int& iAgentID) {
    vector<double> vecDbCapital;

    // Get the capital of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    vecDbCapital.push_back(ptrFirmOfActingAgent->getDbCapital());

    // Get the capital of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            vecDbCapital.push_back(firmPtr->getDbCapital());
        }
    }

    // Get the capital of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            vecDbCapital.push_back(firmPtr->getDbCapital());
        }
    }

    return vecDbCapital;
}

vector<double> Simulator::get_market_overlap_representation() {
    // Given that this information is static across a simulation, we want to only calculate the result once per
    // simulation. Also note that although this is a symmetric matrix, we still calculate the full matrix because
    // the Tableau visualization of the market overlap requires it. (It also will not be symmetric if we allow
    // markets to vary in their number of required capabilities.)
    if (currentSimulationHistoryPtr->vecOfVecMarketOverlapMatrix.empty()) {
        for (Market market1 : economy.get_vec_markets()) {
            vector<double> vecOverlap;
            for (Market market2 : economy.get_vec_markets()) {
                double dbOverlap = MiscUtils::get_percentage_overlap(market1.get_vec_capabilities(),
                                                                     market2.get_vec_capabilities());

                // This percentage overlap represents the percentage of all possible capabilities in the entire economy held by both markets.
                // To convert this to the percent of Market A's capabilities that market B also requires, we must scale this percentage as follows:

                auto defaultEconomyParameters = this->simulatorConfigs["default_economy_parameters"];
                int iNumPossibleCapabilities = defaultEconomyParameters["possible_capabilities"];
                int iCapabilitiesPerMarket = defaultEconomyParameters["capabilities_per_market"];

                double dbCommonCapabilities = iNumPossibleCapabilities * dbOverlap;
                double dbTrueOverlap = dbCommonCapabilities / iCapabilitiesPerMarket;

                vecOverlap.push_back(dbTrueOverlap);
            }
            currentSimulationHistoryPtr->vecOfVecMarketOverlapMatrix.push_back(vecOverlap);
        }
    }
    return MiscUtils::flatten(currentSimulationHistoryPtr->vecOfVecMarketOverlapMatrix);
}

vector<double> Simulator::get_variable_cost_representation(const int& iAgentID) {
    // TODO: Implement a binary mask so that the AI agent can only know that variable cost of a firm-market combination
    //  if the given firm has been present in the given market during the current simulation. For now, the AI agent
    //  can see all variable costs.
    vector<double> vecDbVarCosts;

    // Get the variable costs for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        auto pair = std::make_pair(ptrFirmOfActingAgent->getFirmID(), iMarketID);
        double dbVarCost = dataCache.mapFirmMarketComboToVarCost[pair];
        vecDbVarCosts.push_back(dbVarCost);
    }

    // Get the variable costs of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbVarCost = dataCache.mapFirmMarketComboToVarCost[pair];
                vecDbVarCosts.push_back(dbVarCost);
            }
        }
    }

    // Get the variable costs of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbVarCost = dataCache.mapFirmMarketComboToVarCost[pair];
                vecDbVarCosts.push_back(dbVarCost);
            }
        }
    }

    return vecDbVarCosts;
}

vector<double> Simulator::get_fixed_cost_representation(const int& iAgentID) {
    vector<double> vecDbFixedCosts;

    // Get the fixed costs for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        auto pair = std::make_pair(ptrFirmOfActingAgent->getFirmID(), iMarketID);
        double dbFixedCost = dataCache.mapFirmMarketComboToFixedCost[pair];
        vecDbFixedCosts.push_back(dbFixedCost);
    }

    // Get the fixed costs of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbFixedCost = dataCache.mapFirmMarketComboToFixedCost[pair];
                vecDbFixedCosts.push_back(dbFixedCost);
            }
        }
    }

    // Get the fixed costs of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbFixedCost = dataCache.mapFirmMarketComboToFixedCost[pair];
                vecDbFixedCosts.push_back(dbFixedCost);
            }
        }
    }

    return vecDbFixedCosts;
}

vector<double> Simulator::get_market_portfolio_representation(const int& iAgentID) {
    // Note that while market presence is binary, we use double values here to make the market portfolio
    // representation compatible with the rest of the state observation.

    vector<double> vecDbMarketPortfolioRepresentation;

    // Get the portfolio for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        if (ptrFirmOfActingAgent->is_in_market(economy.get_market_by_ID(iMarketID))) {
            vecDbMarketPortfolioRepresentation.push_back(1.0);
        }
        else {
            vecDbMarketPortfolioRepresentation.push_back(0.0);
        }
    }

    // Get the portfolio of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                if (firmPtr->is_in_market(economy.get_market_by_ID(iMarketID))) {
                    vecDbMarketPortfolioRepresentation.push_back(1.0);
                }
                else {
                    vecDbMarketPortfolioRepresentation.push_back(0.0);
                }
            }
        }
    }

    // Get the portfolio of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                if (firmPtr->is_in_market(economy.get_market_by_ID(iMarketID))) {
                    vecDbMarketPortfolioRepresentation.push_back(1.0);
                }
                else {
                    vecDbMarketPortfolioRepresentation.push_back(0.0);
                }
            }
        }
    }

    return vecDbMarketPortfolioRepresentation;
}

vector<double> Simulator::get_entry_cost_representation(const int& iAgentID) {
    vector<double> vecDbEntryCosts;

    // Get the entry costs for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        auto pair = std::make_pair(ptrFirmOfActingAgent->getFirmID(), iMarketID);
        double dbEntryCost = dataCache.mapFirmMarketComboToEntryCost[pair];
        vecDbEntryCosts.push_back(dbEntryCost);
    }

    // Get the entry costs of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbEntryCost = dataCache.mapFirmMarketComboToEntryCost[pair];
                vecDbEntryCosts.push_back(dbEntryCost);
            }
        }
    }

    // Get the entry costs of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbEntryCost = dataCache.mapFirmMarketComboToEntryCost[pair];
                vecDbEntryCosts.push_back(dbEntryCost);
            }
        }
    }

    return vecDbEntryCosts;
}

vector<double> Simulator::get_demand_intercept_representation() {
    vector<double> vecDbDemandIntercepts;
    for (auto market : economy.get_vec_markets()) {
        vecDbDemandIntercepts.push_back(market.getDbDemandIntercept());
    }
    return vecDbDemandIntercepts;
}

vector<double> Simulator::get_demand_slope_representation() {
    vector<double> vecDbDemandSlopes;
    for (auto market : economy.get_vec_markets()) {
        vecDbDemandSlopes.push_back(market.getDbDemandSlope());
    }
    return vecDbDemandSlopes;
}

vector<double> Simulator::get_quantity_representation(const int& iAgentID) {
    vector<double> vecDbQuantities;

    // Get the production quantities for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        auto pair = std::make_pair(ptrFirmOfActingAgent->getFirmID(), iMarketID);
        double dbQty = dataCache.mapFirmMarketComboToQtyProduced[pair];
        vecDbQuantities.push_back(dbQty);
    }

    // Get the production quantities of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbQty = dataCache.mapFirmMarketComboToQtyProduced[pair];
                vecDbQuantities.push_back(dbQty);
            }
        }
    }

    // Get the production quantities of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbQty = dataCache.mapFirmMarketComboToQtyProduced[pair];
                vecDbQuantities.push_back(dbQty);
            }
        }
    }

    return vecDbQuantities;
}


vector<double> Simulator::get_price_representation(const int& iAgentID) {
    vector<double> vecDbPrices;

    // Get the prices for the firm of the acting agent
    auto ptrFirmOfActingAgent = get_firm_ptr_from_agent_id(iAgentID);
    for (int iMarketID : get_set_market_IDs()) {
        auto pair = std::make_pair(ptrFirmOfActingAgent->getFirmID(), iMarketID);
        double dbPrice = dataCache.mapFirmMarketComboToPrice[pair];
        vecDbPrices.push_back(dbPrice);
    }

    // Get the prices of any other AI agents
    for (auto entry : mapAgentIDToAgentPtr) {
        // Skip the acting agent (already accounted for above)
        if (entry.first == iAgentID)
            continue;
        if (is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbPrice = dataCache.mapFirmMarketComboToPrice[pair];
                vecDbPrices.push_back(dbPrice);
            }
        }
    }

    // Get the prices of all control agents
    for (auto entry : mapAgentIDToAgentPtr) {
        if (!is_ai_agent(entry.first)) {
            auto firmPtr = get_firm_ptr_from_agent_id(entry.first);
            for (int iMarketID : get_set_market_IDs()) {
                auto pair = std::make_pair(firmPtr->getFirmID(), iMarketID);
                double dbPrice = dataCache.mapFirmMarketComboToPrice[pair];
                vecDbPrices.push_back(dbPrice);
            }
        }
    }
    return vecDbPrices;
}

// Generates reward for the specified RL agent as its change in capital since the last time it acted
double Simulator::generate_reward(const int& iAgentID) {
    auto firmPtr = get_firm_ptr_from_agent_id(iAgentID);
    double dbCurrentCapital = firmPtr->getDbCapital();
    double dbPreviousCapital = mapAIAgentIDToCapitalAtLastTurn[iAgentID];
    int iPreviousMicroTimeStep = mapAIAgentIDToMicroTimeStepOfLastTurn[iAgentID];
    double dbCapitalChange = dbCurrentCapital - dbPreviousCapital;
    if (iCurrentMicroTimeStep == iPreviousMicroTimeStep) {
        throw std::runtime_error("generate_reward called with current and previous time steps equal");
    }
    double dbAverageCapitalChange = dbCapitalChange / (iCurrentMicroTimeStep - iPreviousMicroTimeStep);
    return dbAverageCapitalChange;
}

int Simulator::get_next_AI_agent_index() {
    int iAIAgentNumber = iNumAITurns % iNumAIAgents; // Which, of all the AI agents, is acting?
    // For example, if there are two AI agents,
    // this will either be 0 or 1. iNumAITurns
    // is the number of AI turns that have taken
    // place thus far in the simulation.

    int iTurnsSearched = 0; // How many AI agent turns we have checked so far
    for (int iAgentID : vecAgentTurnOrder) {
        if (is_ai_agent(iAgentID)) {
            if (iTurnsSearched == iAIAgentNumber) {
                return iAgentID;
            }
            iTurnsSearched++;
        }
    }

    // Shouldn't reach this part of the code
    cerr << "Error in Simulator::get_next_AI_agent_index()" << endl;
    throw std::exception();
}

int Simulator::get_num_AI_agents() {
    return iNumAIAgents;
}

int Simulator::get_num_total_agents() {
    return mapAgentIDToAgentPtr.size();
}

bool Simulator::at_beginning_of_macro_step() {
    return iCurrentMicroTimeStep % get_micro_steps_per_macro_step() == 0;
}