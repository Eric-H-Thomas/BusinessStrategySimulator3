//
// Created by Eric Thomas on 10/3/23.
//

#pragma once
#include <map>
#include <set>

using std::map;
using std::pair;

/*
THE PURPOSE OF THE DATA CACHE IS TO STORE INFORMATION THAT CORRESPONDS TO FIRM-MARKET COMBINATIONS RATHER THAN
SOLELY TO A FIRM OR A MARKET. DO NOT STORE INFORMATION HERE THAT IS ONLY SPECIFIC TO A FIRM OR A MARKET RATHER THAN
A FIRM-MARKET COMBINATION.

This class stores current values for
    - the revenue               in the most recent micro time step for each firm-market combination
    - the fixed cost            in the most recent micro time step for each firm-market combination
    - the variable cost         in the most recent micro time step for each firm-market combination
    - the entry cost            in the most recent micro time step for each firm-market combination
    - the quantity produced     in the most recent micro time step for each firm-market combination
    - the price per unit        in the most recent micro time step for each firm-market combination

Note: While the Cournot model--which is the only economic model implemented in this simulator as of 12.20.23--
requires one price per market, we record price changes according to firm-market combinations in case we want to
change the simulator to allow for intra-market price variation.
*/

class DataCache {
public:
    // Default constructor
    DataCache();

    bool bInitialized = false;
    map<pair<int, int>, double> mapFirmMarketComboToRevenue;
    map<pair<int, int>, double> mapFirmMarketComboToFixedCost;
    map<pair<int, int>, double> mapFirmMarketComboToVarCost;
    map<pair<int, int>, double> mapFirmMarketComboToEntryCost;
    map<pair<int, int>, double> mapFirmMarketComboToQtyProduced;
    map<pair<int, int>, double> mapFirmMarketComboToPrice;
};