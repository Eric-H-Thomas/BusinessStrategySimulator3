//
// Created by Eric Thomas on 9/7/23.
//

#pragma once

#include <vector>
#include <map>
#include <string>
#include <set>

using std::map;
using std::vector;
using std::pair;
using std::string;

// To keep the history as lightweight as possible, we will record changes in the states of simulator objects
// rather than recording values for such objects at every time step. For example, rather than recording that firm A was
// present in market X for time steps 1, 2 and 3 and not present for steps 4, 5, and 6, we will just record an entrance
// at time step 1 and an exit at time step 4.

// Predeclare the structs so that we can keep the SimulationHistory info at the top of the file
struct CapitalChange; struct RevenueChange; struct FixedCostChange; struct EntryCostChange;
struct ProductionQuantityChange; struct PriceChange; struct MarketPresenceChange;

class SimulationHistory {
public:
    map<int, int>                     mapAgentToFirm;
    map<int, string>                  mapFirmToAgentDescription;
    map<int, double>                  mapFirmStartingCapital;
    map<int, double>                  mapMarketMaximumEntryCost;
    map<pair<int, int>, double>       mapFirmMarketComboToVarCost;
    vector<vector<double>>            vecOfVecMarketOverlapMatrix;
    vector<CapitalChange>             vecCapitalChanges;
    vector<RevenueChange>             vecRevenueChanges;
    vector<FixedCostChange>           vecFixedCostChanges;
    vector<EntryCostChange>           vecEntryCostChanges;
    vector<ProductionQuantityChange>  vecProductionQtyChanges;
    vector<PriceChange>               vecPriceChanges;
    vector<MarketPresenceChange>      vecMarketPresenceChanges;

    SimulationHistory(const map<int, int>& mapAgentToFirm, const map<int, string>& mapFirmToAgentDescription,
                      const map<int, double>& mapFirmStartingCapital,
                      const map<int, double>& mapMarketMaximumEntryCost);

    void record_market_presence_change(int iMicroTimeStep, bool bPresent, int iFirmID, int iMarketID);
    void record_capital_change(int iMicroTimeStep, int iFirmID, double dbNewCapitalQty);
    void record_revenue_change(int iMicroTimeStep, double dbNewRevenueAmount, int iFirmID, int iMarketID);
    void record_fixed_cost_change(int iMicroTimeStep, double dbNewFixedCost, int iFirmID, int iMarketID);
    void record_entry_cost_change(int iMicroTimeStep, double dbNewEntryCost, int iFirmID, int iMarketID);
    void record_production_quantity_change(int iMicroTimeStep, double dbNewProductionQty, int iFirmID, int iMarketID);
    void record_price_change(int iMicroTimeStep, double dbNewPrice, int iFirmID, int iMarketID);
    void record_bankruptcy(int iMicroTimeStep, int iFirmID, std::set<int> setMarketPortfolio);
};

struct CapitalChange {
    int iMicroTimeStep;
    int iFirmID;
    double dbNewCapitalQty;
};

struct RevenueChange {
    int iMicroTimeStep;
    double dbNewRevenueAmount;
    int iFirmID;
    int iMarketID;
};

struct FixedCostChange {
    int iMicroTimeStep;
    double dbNewFixedCost;
    int iFirmID;
    int iMarketID;
};

struct EntryCostChange {
    int iMicroTimeStep;
    double dbNewEntryCost;
    int iFirmID;
    int iMarketID;
};

struct ProductionQuantityChange {
    int iMicroTimeStep;
    double dbNewProductionQty;
    int iFirmID;
    int iMarketID;
};

struct PriceChange {
    int iMicroTimeStep;
    double dbNewPrice;
    int iFirmID;
    int iMarketID;
};

struct MarketPresenceChange {
    int iMicroTimeStep;
    bool bPresent;
    int iFirmID;
    int iMarketID;
};