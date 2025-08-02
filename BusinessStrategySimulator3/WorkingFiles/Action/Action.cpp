//
// Created by Eric Thomas on 9/14/23.
//

#include "Action.h"

#define NOT_APPLICABLE -1

Action Action::generate_none_action(int iAgentID) {
    Action action;
    action.iAgentID = iAgentID;
    action.enumActionType = ActionType::enumNoneAction;
    action.iMarketID = NOT_APPLICABLE;
    action.iMicroTimeStep = NOT_APPLICABLE;
    return action;
}

// Constructors
Action::Action() {}
Action::Action(int iAgentId, ActionType enumActionType, int iMarketId, int iMicroTimeStep) : iAgentID(iAgentId),
                                                                                             enumActionType(enumActionType),
                                                                                             iMarketID(iMarketId),
                                                                                             iMicroTimeStep(iMicroTimeStep) {}