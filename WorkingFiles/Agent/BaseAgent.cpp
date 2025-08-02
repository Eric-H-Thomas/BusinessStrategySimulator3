//
// Created by Eric Thomas on 12/21/23.
//

#include "BaseAgent.h"

int BaseAgent::get_agent_ID() const {
    return iAgentID;
}

string BaseAgent::get_path_to_agent() const {
    return strPathToAgent;
}