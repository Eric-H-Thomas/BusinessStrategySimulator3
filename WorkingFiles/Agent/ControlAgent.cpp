//
// Created by Eric Thomas on 8/17/23.
//

#include "ControlAgent.h"
#include "../Utils/StringUtils.h"
#include <vector>
#include <iostream>
using std::endl;
using std::vector;
using std::cerr;

#define NOT_YET_SET -1

ControlAgent::ControlAgent(const int& iAgentID, const string& strEntryPolicy,
                           const string& strExitPolicy, const string& strProductionPolicy,
                           const double& dbEntryActionLikelihood,
                           const double& dbExitActionLikelihood, const double& dbNoneActionLikelihood) {
    // Set the agent type
    this->enumAgentType = AgentType::Control;

    // Set the entry policy
    if (StringUtils::equalsIgnoreCase(strEntryPolicy, "all"))
        this->enumEntryPolicy = EntryPolicy::All;
    if (StringUtils::equalsIgnoreCaseAndIgnoreUnderscores(strEntryPolicy, "highest_overlap"))
        this->enumEntryPolicy = EntryPolicy::HighestOverlap;

    // Set the exit policy
    if (StringUtils::equalsIgnoreCase(strExitPolicy, "all"))
        this->enumExitPolicy = ExitPolicy::All;
    if (StringUtils::equalsIgnoreCase(strExitPolicy, "loss")) {
        this->enumExitPolicy = ExitPolicy::Loss;
    }

    // Set the production policy
    if (StringUtils::equalsIgnoreCase(strProductionPolicy, "cournot")) {
        this->enumProductionPolicy = ProductionPolicy::Cournot;
    }

    // Set the remaining agent hyperparameters
    this->iAgentID = iAgentID;
    this->dbEntryActionLikelihood = dbEntryActionLikelihood;
    this->dbExitActionLikelihood = dbExitActionLikelihood;
    this->dbNoneActionLikelihood = dbNoneActionLikelihood;
    this->iFirmAssignment = NOT_YET_SET;
}

vector<double> ControlAgent::get_action_likelihood_vector() const {
    return { dbEntryActionLikelihood, dbExitActionLikelihood, dbNoneActionLikelihood };
}

// Getters
EntryPolicy         ControlAgent::get_enum_entry_policy()       const { return enumEntryPolicy; }
ExitPolicy          ControlAgent::get_enum_exit_policy()        const { return enumExitPolicy; }
ProductionPolicy    ControlAgent::get_enum_production_policy()  const { return enumProductionPolicy; }

string ControlAgent::to_string() const {
    // Agent ID number
    string output = "ID:";
    output += std::to_string(iAgentID);

    // Agent type
    output += "__";
    output += "Agent type: ControlAgent";

    // Entry policy
    output += "__";
    output += "Entry Policy:";
    if (enumEntryPolicy == EntryPolicy::All) {
        output += "All";
    }
    else if (enumEntryPolicy == EntryPolicy::HighestOverlap) {
        output += "HighestOverlap";
    }
    else {
        throw std::runtime_error("Haha nice try bud. Entry policy not yet configured in control agent toString method");
    }

    // Exit policy
    output += "__";
    output += "Exit Policy:";
    if (enumExitPolicy == ExitPolicy::All) {
        output += "All";
    }
    else if (enumExitPolicy == ExitPolicy::Loss) {
        output += "Loss";
    }
    else {
        throw std::runtime_error("Haha nice try bud. Exit policy not yet configured in control agent toString method");
    }

    // Production policy
    output += "__";
    output += "Production Policy:";
    if (enumProductionPolicy == ProductionPolicy::Cournot) {
        output += "Cournot";
    }
    else {
        throw std::runtime_error("Haha nice try bud. Production policy not yet configured in control agent toString method");
    }

    // Entry action likelihood
    output += "__";
    output += "Entry action likelihood:";
    output += std::to_string(dbEntryActionLikelihood);

    // Exit action likelihood
    output += "__";
    output += "Exit action likelihood:";
    output += std::to_string(dbExitActionLikelihood);

    // None action likelihood
    output += "__";
    output += "None action likelihood:";
    output += std::to_string(dbNoneActionLikelihood);

    return output;
}
