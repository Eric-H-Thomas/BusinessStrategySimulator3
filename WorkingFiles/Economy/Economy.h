//
// Created by Eric Thomas on 8/17/23.
//

#pragma once
#include <iostream>
#include <ostream>
#include <set>
#include "../Market/Market.h"

using std::vector;
using std::set;

class Economy {
public:
    // Constructors
    Economy();
    Economy(int iPossibleCapabilities, int iCapabilitiesPerMarket, int iNumMarketClusters, vector<int> vecClusterMeans,
            vector<int> vecClusterSDs, vector<int> vecMarketsPerCluster, double dbMarketEntryCostMax,
            double dbMarketEntryCostMin);

    // Getters
    [[nodiscard]] const Market&           get_market_by_ID(int iMarketID)   const;
    [[nodiscard]] const vector<Market>&   get_vec_markets()                 const;
    [[nodiscard]] set<int>                get_set_market_IDs()              const;
    [[nodiscard]] const vector<double>&   get_vec_capability_costs()        const;
    [[nodiscard]] const vector<int>&      get_vec_cluster_means()           const;
    [[nodiscard]] const vector<int>&      get_vec_cluster_SDs()             const;
    [[nodiscard]] const vector<int>&      get_vec_markets_per_cluster()     const;
    [[nodiscard]] int                     get_total_markets()               const;
    [[nodiscard]] int                     get_num_market_clusters()         const;
    [[nodiscard]] int                     get_num_possible_capabilities()   const;
    [[nodiscard]] int                     get_num_capabilities_per_market() const;

    // Miscellaneous
    void add_market(const Market& market);
    void clear_markets();

private:
    int iPossibleCapabilities;
    int iCapabilitiesPerMarket;
    int iNumMarketClusters{};
    vector<Market> vecMarkets;
    vector<int> vecClusterMeans;
    vector<int> vecClusterSDs;
    vector<int> vecMarketsPerCluster;
    vector<double> vecCapabilityCosts;
};