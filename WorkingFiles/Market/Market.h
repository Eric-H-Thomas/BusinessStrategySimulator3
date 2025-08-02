//
// Created by Eric Thomas on 9/5/23.
//

#pragma once
#include <vector>

using std::vector;

class Market {
public:
    Market();
    Market(int iMarketID, double dbFixedCostAsPercentageOfEntryCost, double dbExitCostAsPercentageOfEntryCost,
           double dbDemandIntercept, double dbDemandSlope,
           const vector<int>& vecCapabilities);
    const vector<int>& get_vec_capabilities() const;
    const int& get_market_id() const;
    const double& getDbDemandIntercept() const;
    const double& getDbDemandSlope() const;
    const double& getExitCostAsPercentageOfEntryCost() const;
    const double& getFixedCostAsPercentageOfEntryCost() const;

    // Comparison operator to allow markets to be placed in ordered data structures
    bool operator<(const Market& other) const;

private:
    int    iMarketID;
    double dbFixedCostAsPercentageOfEntryCost;
    double dbExitCostAsPercentageOfEntryCost;
    double dbDemandIntercept;
    double dbDemandSlope;
    vector<int> vecCapabilities;
};