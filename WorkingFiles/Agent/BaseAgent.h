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
    [[nodiscard]] virtual string to_string() const = 0;
    [[nodiscard]] virtual ProductionPolicy get_enum_production_policy() const = 0;
    [[nodiscard]] int get_agent_ID() const;
    [[maybe_unused]] [[nodiscard]] const string& get_path_to_agent() const;
    virtual ~BaseAgent() = default;

protected:
    int iAgentID;
    string strPathToAgent; // Not needed for control agents
};