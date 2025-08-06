//
// Created by Eric Thomas on 9/7/23.
//

#include "SimulationHistory.h"

SimulationHistory::SimulationHistory(const map<int, int>& mapAgentToFirm,
                                     const map<int, string>& mapFirmToAgentDescription,
                                     const map<int, double>& mapFirmStartingCapital,
                                     const map<int, double>& mapMarketMaximumEntryCost) :
        mapAgentToFirm(mapAgentToFirm),
        mapFirmToAgentDescription(mapFirmToAgentDescription),
        mapFirmStartingCapital(mapFirmStartingCapital),
        mapMarketMaximumEntryCost(mapMarketMaximumEntryCost) {
}

void SimulationHistory::record_bankruptcy(int iMicroTimeStep, int iFirmID, const std::set<int>& setMarketPortfolio) {
    // Record bankruptcy in the simulation history
    for (int iMarketID : setMarketPortfolio) {
        record_market_presence_change(iMicroTimeStep, false, iFirmID, iMarketID);
        record_revenue_change(iMicroTimeStep, 0.0, iFirmID, iMarketID);
        record_fixed_cost_change(iMicroTimeStep, 0.0, iFirmID, iMarketID);
        record_entry_cost_change(iMicroTimeStep, 0.0, iFirmID, iMarketID);
        record_production_quantity_change(iMicroTimeStep, 0.0, iFirmID, iMarketID);
        record_price_change(iMicroTimeStep, 0.0, iFirmID, iMarketID);
    }

    record_capital_change(iMicroTimeStep, iFirmID, -1e-9);
}

void SimulationHistory::record_capital_change(int iMicroTimeStep, int iFirmID, double dbNewCapitalQty) {
    vecCapitalChanges.emplace_back(CapitalChange{iMicroTimeStep, iFirmID, dbNewCapitalQty});
}

void SimulationHistory::record_revenue_change(int iMicroTimeStep, double dbNewRevenueAmount, int iFirmID, int iMarketID) {
    vecRevenueChanges.emplace_back(RevenueChange{iMicroTimeStep, dbNewRevenueAmount, iFirmID, iMarketID});
}

void SimulationHistory::record_fixed_cost_change(int iMicroTimeStep, double dbNewFixedCost, int iFirmID, int iMarketID) {
    vecFixedCostChanges.emplace_back(FixedCostChange{iMicroTimeStep, dbNewFixedCost, iFirmID, iMarketID});
}

void SimulationHistory::record_entry_cost_change(int iMicroTimeStep, double dbNewEntryCost, int iFirmID, int iMarketID) {
    vecEntryCostChanges.emplace_back(EntryCostChange{iMicroTimeStep, dbNewEntryCost, iFirmID, iMarketID});
}

void SimulationHistory::record_production_quantity_change(int iMicroTimeStep, double dbNewProductionQty, int iFirmID, int iMarketID) {
    vecProductionQtyChanges.emplace_back(ProductionQuantityChange{iMicroTimeStep, dbNewProductionQty, iFirmID, iMarketID});
}

void SimulationHistory::record_price_change(int iMicroTimeStep, double dbNewPrice, int iFirmID, int iMarketID) {
    vecPriceChanges.emplace_back(PriceChange{iMicroTimeStep, dbNewPrice, iFirmID, iMarketID});
}

void SimulationHistory::record_market_presence_change(int iMicroTimeStep, bool bPresent, int iFirmID, int iMarketID) {
    vecMarketPresenceChanges.emplace_back(MarketPresenceChange{ iMicroTimeStep, bPresent, iFirmID, iMarketID });
}