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
    const Market&           get_market_by_ID(int iMarketID)   const;
    const vector<Market>&   get_vec_markets()                 const;
    set<int>                get_set_market_IDs()              const;
    const vector<double>&   get_vec_capability_costs()        const;
    const vector<int>&      get_vec_cluster_means()           const;
    const vector<int>&      get_vec_cluster_SDs()             const;
    const vector<int>&      get_vec_markets_per_cluster()     const;
    int                     get_total_markets()               const;
    int                     get_num_market_clusters()         const;
    int                     get_num_possible_capabilities()   const;
    int                     get_num_capabilities_per_market() const;

    // Miscellaneous
    void add_market(Market market);

private:
    int iPossibleCapabilities;
    int iCapabilitiesPerMarket;
    int iNumMarketClusters;
    vector<Market> vecMarkets;
    vector<int> vecClusterMeans;
    vector<int> vecClusterSDs;
    vector<int> vecMarketsPerCluster;
    vector<double> vecCapabilityCosts;
};