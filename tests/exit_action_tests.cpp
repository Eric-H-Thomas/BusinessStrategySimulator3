#include <cassert>
#include <map>
#include <string>
#include <vector>
#include <iostream>

#include "JSONReader/json.h"
#include "Agent/ControlAgent.h"
#include "History/SimulationHistory.h"
#include "Market/Market.h"
#include "Economy/Economy.h"
#include "Firm/Firm.h"
#include "Action/Action.h"

#define private public
#include "Simulator/Simulator.h"
#undef private

using namespace std;

void test_exit_executes_with_sufficient_capital() {
    Simulator sim;
    sim.bVerbose = false;

    Firm firm(1, 100.0, 3);
    ControlAgent agent(1, "All", "All", "Cournot", 0.0, 0.0, 100.0);
    agent.iFirmAssignment = 1;
    sim.mapFirmIDToFirmPtr[1] = &firm;
    sim.mapAgentIDToAgentPtr[1] = &agent;

    vector<int> vecCapabilities{1, 0, 0};
    Market market(1, 0.0, 0.0, 0.0, 0.0, vecCapabilities);
    Economy economy(3, 3, 1, vector<int>{1}, vector<int>{1}, vector<int>{1}, 100.0, 50.0);
    economy.add_market(market);
    sim.economy = economy;

    pair<int, int> pairFM(1, 1);
    sim.dataCache.mapFirmMarketComboToExitCost[pairFM] = 30.0;
    sim.dataCache.mapFirmMarketComboToEntryCost[pairFM] = 0.0;

    map<int, int> mapAgentToFirm{{1, 1}};
    map<int, string> mapFirmToAgentDesc{{1, ""}};
    map<int, double> mapFirmStartingCapital{{1, 100.0}};
    map<int, double> mapMarketMaxEntryCost{{1, 0.0}};
    SimulationHistory history(mapAgentToFirm, mapFirmToAgentDesc,
                              mapFirmStartingCapital, mapMarketMaxEntryCost);
    sim.currentSimulationHistoryPtr = &history;

    firm.add_market_to_portfolio(1);
    firm.add_market_capabilities_to_firm_capabilities(market);

    map<int, double> mapCapitalChange{{1, 0.0}};
    Action exitAction(1, ActionType::enumExitAction, 1);
    sim.execute_exit_action(exitAction, &mapCapitalChange);

    assert(firm.getDbCapital() == 70.0);
    assert(mapCapitalChange[1] == -30.0);
    assert(!firm.is_in_market(market));
}

void test_exit_skipped_when_capital_insufficient() {
    Simulator sim;
    sim.bVerbose = false;

    Firm firm(1, 20.0, 3);
    ControlAgent agent(1, "All", "All", "Cournot", 0.0, 0.0, 100.0);
    agent.iFirmAssignment = 1;
    sim.mapFirmIDToFirmPtr[1] = &firm;
    sim.mapAgentIDToAgentPtr[1] = &agent;

    vector<int> vecCapabilities{1, 0, 0};
    Market market(1, 0.0, 0.0, 0.0, 0.0, vecCapabilities);
    Economy economy(3, 3, 1, vector<int>{1}, vector<int>{1}, vector<int>{1}, 100.0, 50.0);
    economy.add_market(market);
    sim.economy = economy;

    pair<int, int> pairFM(1, 1);
    sim.dataCache.mapFirmMarketComboToExitCost[pairFM] = 30.0;
    sim.dataCache.mapFirmMarketComboToEntryCost[pairFM] = 0.0;

    map<int, int> mapAgentToFirm{{1, 1}};
    map<int, string> mapFirmToAgentDesc{{1, ""}};
    map<int, double> mapFirmStartingCapital{{1, 20.0}};
    map<int, double> mapMarketMaxEntryCost{{1, 0.0}};
    SimulationHistory history(mapAgentToFirm, mapFirmToAgentDesc,
                              mapFirmStartingCapital, mapMarketMaxEntryCost);
    sim.currentSimulationHistoryPtr = &history;

    firm.add_market_to_portfolio(1);
    firm.add_market_capabilities_to_firm_capabilities(market);

    map<int, double> mapCapitalChange{{1, 0.0}};
    Action exitAction(1, ActionType::enumExitAction, 1);
    sim.execute_exit_action(exitAction, &mapCapitalChange);

    assert(firm.getDbCapital() == 20.0);
    assert(mapCapitalChange[1] == 0.0);
    assert(firm.is_in_market(market));
}

int main() {
    test_exit_executes_with_sufficient_capital();
    test_exit_skipped_when_capital_insufficient();
    cout << "All tests passed\n";
    return 0;
}

