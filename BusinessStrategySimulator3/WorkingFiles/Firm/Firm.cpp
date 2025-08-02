//
// Created by Eric Thomas on 8/31/23.
//

#include "Firm.h"

Firm::Firm() = default;

Firm::Firm(int iFirmID, double dbStartingCapital, int iPossibleCapabilities) {
    this->iFirmID = iFirmID;
    this->dbCapital = dbStartingCapital;
    this->vecCapabilities.resize(iPossibleCapabilities, 0); // Init empty capabilities vector
}

int Firm::getFirmID() const {
    return iFirmID;
}

double Firm::getDbCapital() const {
    return dbCapital;
}

bool Firm::is_in_market(Market market) {
    if (this->setMarketIDs.find(market.get_market_id()) != setMarketIDs.end()) {
        return true;
    }
    return false;
}

int Firm::add_market_to_portfolio(const int& iMarketID) {
    if (!this->setMarketIDs.insert(iMarketID).second) {
        std::cerr << "Unsuccessful market entry" << std::endl;
        return 1;
    }
    return 0;
}

int Firm::add_market_capabilities_to_firm_capabilities(const Market& market) {
    try {
        // Set firm capability vector to (firmCapVec ORed with marketCapVec)
        this->vecCapabilities = MiscUtils::element_wise_logical_or(this->vecCapabilities, market.get_vec_capabilities());
    }
    catch (std::exception e) {
        std::cerr << "Error adding market capabilities to firm capabilities" << std::endl;
        return 1;
    }

    return 0;
}

int Firm::remove_market_capabilities_from_firm_capabilities(const Market& marketToRemove, const Economy& economy) {
    // Begin with an empty capabilities vector
    vector<int> capabilitiesVector(vecCapabilities.size(), 0);

    try {
        // Iterate through every market in the firm's portfolio, counting how many markets require each capability
        for (auto marketID : setMarketIDs) {
            auto market = economy.get_market_by_ID(marketID);
            capabilitiesVector = MiscUtils::vector_addition(capabilitiesVector, market.get_vec_capabilities());
        }

        // Subtract the capability vector of the market we are exiting
        capabilitiesVector = MiscUtils::vector_subtraction(capabilitiesVector, marketToRemove.get_vec_capabilities());

        // Change all positive entries in the vector to a 1
        MiscUtils::set_all_positive_values_to_one(capabilitiesVector);
    }
    catch (std::exception e) {
        std::cerr << "Error removing market capabilities from firm capabilities" << std::endl;
        return 1;
    }

    // Save the result as the new capability vector for the firm
    this->vecCapabilities = capabilitiesVector;

    return 0;
}

int Firm::remove_market_from_portfolio(const int& iMarketID) {
    if (this->setMarketIDs.erase(iMarketID) == 0) {
        std::cerr << "Unsuccessful market exit" << std::endl;
        return 1;
    }
    return 0;
}

Market Firm::choose_market_with_highest_overlap(set<Market> setMarkets) {
    constexpr double EPSILON = 1e-12; // Small epsilon threshold to account for variance in division results
    double dbHighestOverlapValue = -1.0;  // Start below any valid percentage
    std::set<Market> setMarketsWithHighestOverlap;

    for (const auto& market : setMarkets) {
        double dbOverlap = MiscUtils::get_percentage_overlap(
                market.get_vec_capabilities(),
                this->vecCapabilities
        );

        if (dbOverlap > dbHighestOverlapValue + EPSILON) { // Definitely greater than
            dbHighestOverlapValue = dbOverlap;
            setMarketsWithHighestOverlap.clear();
            setMarketsWithHighestOverlap.insert(market);
        }
        else if (std::fabs(dbOverlap - dbHighestOverlapValue) <= EPSILON) { // Equality within epsilon
            setMarketsWithHighestOverlap.insert(market);
        }
    }

    if (setMarketsWithHighestOverlap.empty()) {
        std::cerr << "Error in getting market with highest overlap for firm" << std::endl;
        throw std::runtime_error("no markets to choose from");
    }

    return MiscUtils::choose_random_from_set(setMarketsWithHighestOverlap);
}

const set<int>& Firm::getSetMarketIDs() const {
    return setMarketIDs;
}

void Firm::add_capital(double dbChangeInCapital) {
    this->dbCapital += dbChangeInCapital;
}

const vector<int>& Firm::getVecCapabilities() const {
    return vecCapabilities;
}

void Firm::reset(double dbStartingCapital) {
    // Reset capabilities
    for (size_t i = 0; i < vecCapabilities.size(); i++) {
        vecCapabilities[i] = 0;
    }

    // Reset portfolio
    setMarketIDs.clear();

    // Reset capital
    dbCapital = dbStartingCapital;
}

void Firm::declare_bankruptcy() {
    // Set capital equal to a -1e-9
    dbCapital = -1e-9;

    // Reset capabilities
    for (size_t i = 0; i < vecCapabilities.size(); i++) {
        vecCapabilities[i] = 0;
    }

    // Reset portfolio
    setMarketIDs.clear();
}