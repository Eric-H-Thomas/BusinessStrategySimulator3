//
// Created by Eric Thomas on 8/17/23.
//

#define NOT_YET_SET -1
#include "Economy.h"
#include <random>

Economy::Economy(int iPossibleCapabilities, int iCapabilitiesPerMarket, int iNumMarketClusters, vector<int> vecClusterMeans,
                 vector<int> vecClusterSDs, vector<int> vecMarketsPerCluster, double dbMarketEntryCostMax,
                 double dbMarketEntryCostMin) {

    this->iPossibleCapabilities = iPossibleCapabilities;
    this->iCapabilitiesPerMarket = iCapabilitiesPerMarket;
    this->iNumMarketClusters = iNumMarketClusters;
    this->vecClusterMeans = vecClusterMeans;
    this->vecClusterSDs = vecClusterSDs;
    this->vecMarketsPerCluster = vecMarketsPerCluster;

    // The rest of this constructor initializes the capability costs vector ////////////////////////
    double dbCapCostMin = dbMarketEntryCostMin / iCapabilitiesPerMarket;
    double dbCapCostMax = dbMarketEntryCostMax / iCapabilitiesPerMarket;

    // Create a random number generator engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a uniform distribution in the range [dbCapCostMin, dbCapCostMax) to draw capability costs
    std::uniform_real_distribution<double> capability_cost_dist(dbCapCostMin, dbCapCostMax);

    for (int i = 0; i < iPossibleCapabilities; i++) {
        // Append a randomly generated capability cost to the capability cost vector
        this->vecCapabilityCosts.push_back(capability_cost_dist(gen));
    }
}

Economy::Economy() {
    this->iPossibleCapabilities = NOT_YET_SET;
    this->iCapabilitiesPerMarket = NOT_YET_SET;
}

// Getters
int Economy::get_total_markets()                            const { return vecMarkets.size(); }
int Economy::get_num_market_clusters()                      const { return iNumMarketClusters; }
int Economy::get_num_possible_capabilities()                const { return iPossibleCapabilities; }
int Economy::get_num_capabilities_per_market()              const { return iCapabilitiesPerMarket; }
const vector<int>& Economy::get_vec_cluster_means()         const { return vecClusterMeans; }
const vector<int>& Economy::get_vec_cluster_SDs()           const { return vecClusterSDs; }
const vector<int>& Economy::get_vec_markets_per_cluster()   const { return vecMarketsPerCluster; }
const vector<Market>& Economy::get_vec_markets()            const { return vecMarkets; }
const vector<double>& Economy::get_vec_capability_costs()   const { return vecCapabilityCosts; }

const Market& Economy::get_market_by_ID(int iMarketID) const {
    for (const Market& market : vecMarkets) {
        if (market.get_market_id() == iMarketID) {
            return market;
        }
    }
    std::cerr << "Error getting market by ID" << std::endl;
    throw std::exception();
}

void Economy::add_market(Market market) {
    this->vecMarkets.push_back(market);
}

set<int> Economy::get_set_market_IDs() const {
    set<int> setMarketIDs;
    for (auto market : vecMarkets) {
        setMarketIDs.insert(market.get_market_id());
    }
    return setMarketIDs;
}