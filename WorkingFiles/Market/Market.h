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
    [[nodiscard]] const vector<int>& get_vec_capabilities() const;
    [[nodiscard]] const int& get_market_id() const;
    [[nodiscard]] const double& getDbDemandIntercept() const;
    [[nodiscard]] const double& getDbDemandSlope() const;
    [[nodiscard]] const double& getExitCostAsPercentageOfEntryCost() const;
    [[nodiscard]] const double& getFixedCostAsPercentageOfEntryCost() const;

    // Comparison operator to allow markets to be placed in ordered data structures
    bool operator<(const Market& other) const;

private:
    int    iMarketID{};
    double dbFixedCostAsPercentageOfEntryCost{};
    double dbExitCostAsPercentageOfEntryCost{};
    double dbDemandIntercept{};
    double dbDemandSlope{};
    vector<int> vecCapabilities;
};