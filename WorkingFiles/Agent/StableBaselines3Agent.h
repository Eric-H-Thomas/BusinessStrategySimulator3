//
// Created by Eric Thomas on 12/22/23.
//

#pragma once
#include <string>
#include "BaseAgent.h"
using std::string;

class StableBaselines3Agent : public BaseAgent {
public:
    StableBaselines3Agent(int iAgentID, ProductionPolicy productionPolicy, string strPathToAgent);
    [[nodiscard]] ProductionPolicy get_enum_production_policy() const override;
    [[nodiscard]] string to_string() const override;

private:
    ProductionPolicy enumProductionPolicy;
};




