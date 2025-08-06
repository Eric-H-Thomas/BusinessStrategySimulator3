//
// Created by Eric Thomas on 8/17/23.
//

#pragma once
#include <string>
#include <ostream>
#include <vector>
#include "BaseAgent.h"


using std::string;
enum class EntryPolicy {
    All,
    HighestOverlap
};

enum class ExitPolicy {
    All,
    Loss
};

class ControlAgent : public BaseAgent {
public:
    ControlAgent(const int& iAgentID, const string& strEntryPolicy,
                 const string& strExitPolicy, const string& strProductionPolicy,
                 const double& dbEntryActionLikelihood,
                 const double& dbExitActionLikelihood, const double& dbNoneActionLikelihood);
    [[nodiscard]] std::vector<double> get_action_likelihood_vector() const;
    [[nodiscard]] EntryPolicy get_enum_entry_policy() const;
    [[nodiscard]] ExitPolicy get_enum_exit_policy() const;
    [[nodiscard]] ProductionPolicy get_enum_production_policy() const override;
    [[nodiscard]] string to_string() const override;

private:
    EntryPolicy enumEntryPolicy;
    ExitPolicy enumExitPolicy;
    ProductionPolicy enumProductionPolicy;
    double dbEntryActionLikelihood;
    double dbExitActionLikelihood;
    double dbNoneActionLikelihood;
};
