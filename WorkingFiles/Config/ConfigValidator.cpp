#include "ConfigValidator.h"

#include <fstream>
#include <stdexcept>
#include "../../JSONReader/json.h"

void validate_config(const std::string& strConfigFilePath) {

    std::ifstream file(strConfigFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file at " + strConfigFilePath);
    }

    nlohmann::json config;
    try {
        file >> config;
    }
    catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
    }

    auto require = [](bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    };

    // Simulation parameters
    require(config.contains("simulation_parameters") && config["simulation_parameters"].is_object(),
            "Missing or invalid simulation_parameters section");
    const auto& sim = config["simulation_parameters"];
    require(sim.contains("results_dir") && sim["results_dir"].is_string(),
            "simulation_parameters.results_dir must be a string");
    require(sim.contains("num_sims") && sim["num_sims"].is_number_integer(),
            "simulation_parameters.num_sims must be an integer");
    require(sim.contains("macro_steps_per_sim") && sim["macro_steps_per_sim"].is_number_integer(),
            "simulation_parameters.macro_steps_per_sim must be an integer");
    require(sim.contains("skipped_turns_per_regular_turn") && sim["skipped_turns_per_regular_turn"].is_number_integer(),
            "simulation_parameters.skipped_turns_per_regular_turn must be an integer");
    require(sim["skipped_turns_per_regular_turn"].get<int>() >= 0,
            "simulation_parameters.skipped_turns_per_regular_turn must be non-negative");
    require(sim.contains("generate_master_output") && sim["generate_master_output"].is_boolean(),
            "simulation_parameters.generate_master_output must be a boolean");
    require(sim.contains("verbose") && sim["verbose"].is_boolean(),
            "simulation_parameters.verbose must be a boolean");
    require(sim.contains("randomize_turn_order_within_each_macro_step") &&
            sim["randomize_turn_order_within_each_macro_step"].is_boolean(),
            "simulation_parameters.randomize_turn_order_within_each_macro_step must be a boolean");
    require(sim.contains("randomize_agent_firm_assignment_per_simulation") &&
            sim["randomize_agent_firm_assignment_per_simulation"].is_boolean(),
            "simulation_parameters.randomize_agent_firm_assignment_per_simulation must be a boolean");
    require(sim.contains("randomize_variable_costs_per_simulation") &&
            sim["randomize_variable_costs_per_simulation"].is_boolean(),
            "simulation_parameters.randomize_variable_costs_per_simulation must be a boolean");
    require(sim.contains("randomize_economy_per_simulation") &&
            sim["randomize_economy_per_simulation"].is_boolean(),
            "simulation_parameters.randomize_economy_per_simulation must be a boolean");
    require(sim.contains("randomize_markets_per_simulation") &&
            sim["randomize_markets_per_simulation"].is_boolean(),
            "simulation_parameters.randomize_markets_per_simulation must be a boolean");
    require(sim.contains("fixed_cost_for_existence") &&
            sim["fixed_cost_for_existence"].is_boolean(),
            "simulation_parameters.fixed_cost_for_existence must be a boolean");

    // Control agents
    require(config.contains("control_agents") && config["control_agents"].is_array(),
            "control_agents must be an array");
    for (const auto& agent : config["control_agents"]) {
        require(agent.contains("agent_id") && agent["agent_id"].is_number_integer(),
                "control agent id must be an integer");
        require(agent.contains("entry_policy") && agent["entry_policy"].is_string(),
                "control agent entry_policy must be a string");
        require(agent.contains("exit_policy") && agent["exit_policy"].is_string(),
                "control agent exit_policy must be a string");
        require(agent.contains("production_policy") && agent["production_policy"].is_string(),
                "control agent production_policy must be a string");
        require(agent.contains("entry_action_likelihood") && agent["entry_action_likelihood"].is_number_integer(),
                "control agent entry_action_likelihood must be an integer");
        require(agent.contains("exit_action_likelihood") && agent["exit_action_likelihood"].is_number_integer(),
                "control agent exit_action_likelihood must be an integer");
        require(agent.contains("none_action_likelihood") && agent["none_action_likelihood"].is_number_integer(),
                "control agent none_action_likelihood must be an integer");

        int total = agent["entry_action_likelihood"].get<int>() +
                    agent["exit_action_likelihood"].get<int>() +
                    agent["none_action_likelihood"].get<int>();
        require(total == 100,
                "Action likelihoods for control agent " + std::to_string(agent["agent_id"].get<int>()) +
                " must sum to 100");
    }

    // AI agents
    require(config.contains("ai_agents") && config["ai_agents"].is_array(),
            "ai_agents must be an array");
    for (const auto& agent : config["ai_agents"]) {
        require(agent.contains("agent_id") && agent["agent_id"].is_number_integer(),
                "AI agent id must be an integer");
        require(agent.contains("agent_type") && agent["agent_type"].is_string(),
                "AI agent agent_type must be a string");
        require(agent.contains("production_policy") && agent["production_policy"].is_string(),
                "AI agent production_policy must be a string");
        require(agent.contains("path_to_agent") && agent["path_to_agent"].is_string(),
                "AI agent path_to_agent must be a string");
        require(agent.contains("RL_Algorithm") && agent["RL_Algorithm"].is_string(),
                "AI agent RL_Algorithm must be a string");
    }

    // Economy parameters
    require(config.contains("default_economy_parameters") &&
            config["default_economy_parameters"].is_object(),
            "Missing or invalid default_economy_parameters section");
    const auto& econ = config["default_economy_parameters"];
    require(econ.contains("possible_capabilities") && econ["possible_capabilities"].is_number_integer(),
            "default_economy_parameters.possible_capabilities must be an integer");
    require(econ.contains("capabilities_per_market") && econ["capabilities_per_market"].is_number_integer(),
            "default_economy_parameters.capabilities_per_market must be an integer");
    require(econ.contains("total_markets") && econ["total_markets"].is_number_integer(),
            "default_economy_parameters.total_markets must be an integer");
    require(econ.contains("num_market_clusters") && econ["num_market_clusters"].is_number_integer(),
            "default_economy_parameters.num_market_clusters must be an integer");
    require(econ.contains("cluster_means") && econ["cluster_means"].is_array(),
            "default_economy_parameters.cluster_means must be an array");
    require(econ.contains("cluster_SDs") && econ["cluster_SDs"].is_array(),
            "default_economy_parameters.cluster_SDs must be an array");
    require(econ.contains("markets_per_cluster") && econ["markets_per_cluster"].is_array(),
            "default_economy_parameters.markets_per_cluster must be an array");

    for (const auto& v : econ["cluster_means"]) {
        require(v.is_number_integer(), "cluster_means values must be integers");
    }
    for (const auto& v : econ["cluster_SDs"]) {
        require(v.is_number_integer(), "cluster_SDs values must be integers");
    }
    for (const auto& v : econ["markets_per_cluster"]) {
        require(v.is_number_integer(), "markets_per_cluster values must be integers");
    }

    int numClusters = econ["num_market_clusters"].get<int>();
    require(econ["cluster_means"].size() == static_cast<size_t>(numClusters),
            "cluster_means length must match num_market_clusters");
    require(econ["cluster_SDs"].size() == static_cast<size_t>(numClusters),
            "cluster_SDs length must match num_market_clusters");
    require(econ["markets_per_cluster"].size() == static_cast<size_t>(numClusters),
            "markets_per_cluster length must match num_market_clusters");

    // Firm parameters
    require(config.contains("default_firm_parameters") &&
            config["default_firm_parameters"].is_object(),
            "Missing or invalid default_firm_parameters section");
    const auto& firm = config["default_firm_parameters"];
    require(firm.contains("starting_capital") && firm["starting_capital"].is_number(),
            "default_firm_parameters.starting_capital must be numeric");

    // Market parameters
    require(config.contains("default_market_parameters") &&
            config["default_market_parameters"].is_object(),
            "Missing or invalid default_market_parameters section");
    const auto& market = config["default_market_parameters"];
    require(market.contains("variable_cost_max") && market["variable_cost_max"].is_number(),
            "default_market_parameters.variable_cost_max must be numeric");
    require(market.contains("variable_cost_min") && market["variable_cost_min"].is_number(),
            "default_market_parameters.variable_cost_min must be numeric");
    require(market.contains("fixed_cost_percentage_of_entry") && market["fixed_cost_percentage_of_entry"].is_number_integer(),
            "default_market_parameters.fixed_cost_percentage_of_entry must be an integer");
    require(market.contains("exit_cost_percentage_of_entry") && market["exit_cost_percentage_of_entry"].is_number_integer(),
            "default_market_parameters.exit_cost_percentage_of_entry must be an integer");
    require(market.contains("demand_intercept_max") && market["demand_intercept_max"].is_number_integer(),
            "default_market_parameters.demand_intercept_max must be an integer");
    require(market.contains("demand_intercept_min") && market["demand_intercept_min"].is_number_integer(),
            "default_market_parameters.demand_intercept_min must be an integer");
    require(market.contains("market_entry_cost_max") && market["market_entry_cost_max"].is_number_integer(),
            "default_market_parameters.market_entry_cost_max must be an integer");
    require(market.contains("market_entry_cost_min") && market["market_entry_cost_min"].is_number_integer(),
            "default_market_parameters.market_entry_cost_min must be an integer");
    require(market.contains("product_demand_slope_max") && market["product_demand_slope_max"].is_number(),
            "default_market_parameters.product_demand_slope_max must be numeric");
    require(market.contains("product_demand_slope_min") && market["product_demand_slope_min"].is_number(),
            "default_market_parameters.product_demand_slope_min must be numeric");
}

