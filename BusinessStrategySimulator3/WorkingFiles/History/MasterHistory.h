//
// Created by Eric Thomas on 9/9/23.
//

#pragma once
#include "SimulationHistory.h"
#include <string>
#include <vector>
using std::string;
using std::vector;
class DataRow;

class MasterHistory {
public:
    vector<SimulationHistory*> vecSimulationHistoryPtrs;
    SimulationHistory* getCurrentSimulationHistoryPtr();
    void prepare_data_for_output();
    int generate_master_output();
    int generate_market_overlap_file();
    vector<DataRow> vecDataRows;

    // Data needed to prepare the master output file
    int iMicroStepsPerSim;
    int iNumFirms;
    int iNumMarkets;

    // Data need to help prepare the market overlap file
    int iCapabilitiesPerMarket;

    string strMasterHistoryOutputPath;

private:
    int get_row_number(int iCurrentSim, int iCurrentFirm, int iCurrentMarket, int iCurrentMicroStep);
    void fill_in_capital_info();
    void fill_in_revenue_info();
    void fill_in_market_presence_info();
    void fill_in_variable_cost_info();
    void fill_in_fixed_cost_info();
    void fill_in_entry_cost_info();
    void fill_in_price_info();
    void fill_in_quantity_info();

};

class DataRow {
public:
    int iSim;
    int iMicroTimeStep;
    int iFirmID;
    string strAgentType;
    int iMarketID;
    double dbCapital;
    double dbRevenue;
    double dbFixedCost;
    double dbVarCost;
    double dbEntryCost;
    bool bInMarket;
    double dbPrice;
    double dbQty;
};