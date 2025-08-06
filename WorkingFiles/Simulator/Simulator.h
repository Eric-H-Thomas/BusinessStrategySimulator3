//
// Created by Eric Thomas on 8/17/23.
//

#pragma once

#include <vector>
#include <map>
#include <vector>
#include <string>
#include <limits>
#include "../Agent/ControlAgent.h"
#include "../Agent/StableBaselines3Agent.h"
#include "../Economy/Economy.h"
#include "../Firm/Firm.h"
#include "../History/MasterHistory.h"
#include "../../JSONReader/json.h"
#include "../Utils/MiscUtils.h"
#include "../Market/Market.h"
#include "../DataCache/DataCache.h"

using std::map;
using std::vector;
using std::string;

class Simulator {
public:
    Simulator();
    bool bTrainingMode = false; // Only turn this on from within the Python API
    void load_json_configs(const string& strConfigFilePath);
    void init_master_history();
    void init_simulation_history();
    int init_data_cache(SimulationHistory* pCurrentSimulationHistory);
    void prepare_to_run();
    void set_agent_turn_order();
    // vector<int> get_agent_turn_order();
    int reset();
    void run();
    int get_num_sims() const;
    // int get_macro_steps_per_sim() const;
    int iCurrentMacroTimeStep = 0;
    int iCurrentMicroTimeStep = 0;
    MasterHistory masterHistory;
    bool bVerbose{};
    bool bGenerateMasterOutput{};
    void perform_micro_step_control_agent_or_skip_turn(const int& iActingAgentID);
    void perform_micro_step_ai_agent_turn(const int& iActingAgentID, const int& iAIAgentActionID);
    int perform_micro_step_helper(const vector<Action>& vecActions);
    int get_micro_steps_per_macro_step();
    bool is_ai_agent(const int& iAgentID);
    bool is_bankrupt(const int& iAgentID);

    [[maybe_unused]] vector<double> generate_state_observation(const int& iAgentID);
    vector<double> get_capital_representation(const int& iAgentID);
    vector<double> get_market_overlap_representation();
    vector<double> get_variable_cost_representation(const int& iAgentID);
    vector<double> get_fixed_cost_representation(const int& iAgentID);
    vector<double> get_market_portfolio_representation(const int& iAgentID);
    vector<double> get_entry_cost_representation(const int& iAgentID);
    vector<double> get_demand_intercept_representation();
    vector<double> get_demand_slope_representation();
    vector<double> get_quantity_representation(const int& iAgentID);
    vector<double> get_price_representation(const int& iAgentID);
    double generate_reward(const int& iAgentID);
    int get_next_AI_agent_index();
    [[nodiscard]] int get_num_AI_agents() const;
    int get_num_total_agents();
    bool at_beginning_of_macro_step();
    // int get_num_markets();
    string strResultsDir;

private:
    nlohmann::json simulatorConfigs;
    map<int, BaseAgent*> mapAgentIDToAgentPtr;
    map<int, Firm*> mapFirmIDToFirmPtr;
    Economy economy;
    SimulationHistory* currentSimulationHistoryPtr{};
    DataCache dataCache;
    vector<int> vecAgentTurnOrder;

    // Simulation parameters
    int iNumSims{};
    int iMacroStepsPerSim{};
    double dbSkippedTurnsPerRegularTurn{};
//    bool bFixedCostForExistence;
//    double dbFixedCostForExistence;
    bool bRandomizeTurnOrderWithinEachMacroStep{};
    bool bRandomizeAgentFirmAssignmentPerSimulation{};
    bool bRandomizeVariableCostsPerSimulation{};
    bool bRandomizeEconomyPerSimulation{};
    // Markets are regenerated automatically when the economy is randomized.
    // This flag triggers market regeneration even when the economy remains the same.
    bool bRandomizeMarketsPerSimulation{};


    // Maps to track stats necessary for reward calculations
    map<int, int> mapAIAgentIDToMicroTimeStepOfLastTurn;
    map<int, double> mapAIAgentIDToCapitalAtLastTurn;

    // Helper variables for StableBaselines3 interface
    int iNumAIAgents = 0;
    int iNumAITurns = 0;

    void init_control_agents();
    void init_AI_agents();
    void init_economy();
    void init_markets();
    int reset_economy();
    int reset_markets();
    void set_simulation_parameters();
    // int set_fixed_cost_for_existence();
    void init_firms_for_agents();
    vector<int> create_market_capability_vector(const double& dbMean, const double& dbSD);
    vector<Action> get_actions_for_all_agents_control_agent_turn(const int& iActingAgentID);
    vector<Action> get_actions_for_all_agents_ai_agent_turn(const int& iActingAgentID, const int& iAIAgentActionID);
    Action convert_action_ID_to_action_object(const int& iActingAgentID, const int& iAIAgentActionID);
    int execute_actions(const vector<Action>& vecActions, map<int, double>* pMapFirmIDToCapitalChange);
    int execute_entry_action(const Action& action, map<int, double>* pMapFirmIDToCapitalChange);
    int execute_exit_action(const Action& action, map<int, double>* pMapFirmIDToCapitalChange);
    Action get_agent_action(const ControlAgent& agent);
    static ActionType get_action_type(const ControlAgent& agent);
    Action get_entry_action(const ControlAgent& agent);
    Action get_exit_action(const ControlAgent& agent);
    int distribute_profits(map<int, double>* pMapFirmIDToCapitalChange);
    // Firm* get_firm_ptr_from_agent_ptr(BaseAgent* agentPtr);
    Firm* get_firm_ptr_from_agent_id(const int& iAgentID);
    BaseAgent* get_agent_ptr_from_firm_ID(int iFirmID);
    Firm* get_firm_ptr_from_agent(const ControlAgent& agent);
    set<int> get_set_firm_IDs();
    set<int> get_set_market_IDs();

    set<int> get_firm_IDs_in_market(const Market& market);
    map<int, double> get_map_firm_to_var_cost_for_market(const Market& market);
    double get_average_var_cost_in_market(const Market& market);
    void add_profit_to_firm(double dbProfit, int iFirmID);
    static void shuffle_agent_firm_assignments();
};

