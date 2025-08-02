//
// Created by Eric Thomas on 12/21/23.
//

#pragma once
#include <string>
#include "../Action/Action.h"
using std::string;

enum class AgentType {
    Control,
    StableBaselines3
};

enum class ProductionPolicy {
    Cournot
};

// Base class from which all agent types should be derived
class BaseAgent {
public:
    AgentType enumAgentType;
    int iFirmAssignment;
    virtual string to_string() const = 0;
    virtual ProductionPolicy get_enum_production_policy() const = 0;
    int get_agent_ID() const;
    string get_path_to_agent() const;

protected:
    int iAgentID;
    string strPathToAgent = ""; // Not needed for control agents
};