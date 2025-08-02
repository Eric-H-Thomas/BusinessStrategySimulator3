//
// Created by Eric Thomas on 12/22/23.
//

#pragma once
#include <string>
#include <iostream>
#include "BaseAgent.h"
using std::string;
using std::cerr;
using std::endl;

class StableBaselines3Agent : public BaseAgent {
public:
    StableBaselines3Agent(int iAgentID, ProductionPolicy productionPolicy, string strPathToAgent);
    ProductionPolicy get_enum_production_policy() const override;
    string to_string() const override;

private:
    ProductionPolicy enumProductionPolicy;
};




