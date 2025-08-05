//
// Created by Eric Thomas on 9/14/23.
//

#include "Action.h"

#define NOT_APPLICABLE -1

Action Action::generate_none_action(int iAgentID) {
    return {iAgentID, ActionType::enumNoneAction, NOT_APPLICABLE};
}

// Constructors
Action::Action(int iAgentId, ActionType enumActionType, int iMarketId) :
        iAgentID(iAgentId), enumActionType(enumActionType), iMarketID(iMarketId) {}