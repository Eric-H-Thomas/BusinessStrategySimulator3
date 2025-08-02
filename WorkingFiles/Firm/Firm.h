//
// Created by Eric Thomas on 8/31/23.
//

#pragma once
#include <set>
#include "../Market/Market.h"
#include "../Economy/Economy.h"
#include "../Utils/MiscUtils.h"

using std::set;

class Firm {
public:
    Firm();
    Firm(int iFirmID, double dbStartingCapital, int iPossibleCapabilities);
    void reset(double dbStartingCapital);
    int getFirmID() const;
    double getDbCapital() const;
    const vector<int>& getVecCapabilities() const;
    void add_capital(double dbChangeInCapital);
    int add_market_to_portfolio(const int& iMarketID);
    int add_market_capabilities_to_firm_capabilities(const Market& market);
    int remove_market_capabilities_from_firm_capabilities(const Market& marketToRemove, const Economy& economy);
    int remove_market_from_portfolio(const int& iMarketID);
    bool is_in_market(Market market);
    Market choose_market_with_highest_overlap(set<Market> setMarkets);
    const set<int>& getSetMarketIDs() const;
    void declare_bankruptcy();

private:
    int         iFirmID;
    double      dbCapital;
    vector<int> vecCapabilities;
    set<int>    setMarketIDs;
};