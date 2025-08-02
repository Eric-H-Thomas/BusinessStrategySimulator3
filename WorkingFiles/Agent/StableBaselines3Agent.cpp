//
// Created by Eric Thomas on 12/22/23.
//

#include "StableBaselines3Agent.h"
StableBaselines3Agent::StableBaselines3Agent(int iAgentID, ProductionPolicy productionPolicy, string strPathToAgent) {
    // Set the agent type
    this->enumAgentType = AgentType::StableBaselines3;

    // Set the agent ID
    this->iAgentID = iAgentID;
    this->enumProductionPolicy = productionPolicy;

    // Set the path to the agent
    this->strPathToAgent = strPathToAgent;
}

ProductionPolicy StableBaselines3Agent::get_enum_production_policy() const {
    return this->enumProductionPolicy;
}

string StableBaselines3Agent::to_string() const {
    // Agent ID number
    string output = "ID:";
    output += std::to_string(iAgentID);

    // Agent type
    output += "__";
    output += "Agent type: StableBaselines3";

    // Production policy
    output += "__";
    output += "Production Policy:";
    if (enumProductionPolicy == ProductionPolicy::Cournot) {
        output += "Cournot";
    }
    else {

        cerr << "Haha nice try bud. Production policy not yet configured in control agent toString method" << endl;
        throw std::exception();
    }

    return output;

}