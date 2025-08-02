
//
// Created by Eric Thomas on 9/9/23.
//
#include "MasterHistory.h"
#include <iostream>
#include <fstream>
#include <cmath>

using std::string;
using std::cout;
using std::endl;

SimulationHistory* MasterHistory::getCurrentSimulationHistoryPtr() {
    return vecSimulationHistoryPtrs.back();
}

int MasterHistory::generate_master_output() {
    cout << "Generating master output file." << endl;

    std::ofstream ofStreamMasterOutput;
    string strPath = this->strMasterHistoryOutputPath + "/MasterOutput.csv";
    ofStreamMasterOutput.open(strPath);

    prepare_data_for_output();

    // Create header row
    ofStreamMasterOutput << "" << ",";
    ofStreamMasterOutput << "Sim" << ",";
    ofStreamMasterOutput << "Step" << ",";
    ofStreamMasterOutput << "Firm" << ",";
    ofStreamMasterOutput << "Agent Type" << ",";
    ofStreamMasterOutput << "Market" << ",";
    ofStreamMasterOutput << "Capital" << ",";
    ofStreamMasterOutput << "Rev" << ",";
    ofStreamMasterOutput << "Fix Cost" << ",";
    ofStreamMasterOutput << "Var Cost" << ",";
    ofStreamMasterOutput << "Entry Cost" << ",";
    ofStreamMasterOutput << "In Market" << ",";
    ofStreamMasterOutput << "Price" << ",";
    ofStreamMasterOutput << "Quantity" << ",";
    //ofStreamMasterOutput << "Market Number" << "\n";
    ofStreamMasterOutput << "\n";

    for (int i = 0; i < vecDataRows.size(); i++) {
        auto row = vecDataRows.at(i);
        ofStreamMasterOutput << i << ",";
        ofStreamMasterOutput << row.iSim << ",";
        ofStreamMasterOutput << row.iMicroTimeStep << ",";
        ofStreamMasterOutput << row.iFirmID << ",";
        ofStreamMasterOutput << row.strAgentType << ",";
        ofStreamMasterOutput << row.iMarketID << ",";
        ofStreamMasterOutput << row.dbCapital << ",";
        ofStreamMasterOutput << row.dbRevenue << ",";
        ofStreamMasterOutput << row.dbFixedCost << ",";
        ofStreamMasterOutput << row.dbVarCost << ",";
        ofStreamMasterOutput << row.dbEntryCost << ",";
        ofStreamMasterOutput << row.bInMarket << ",";
        ofStreamMasterOutput << row.dbPrice << ",";
        ofStreamMasterOutput << row.dbQty << ",";
        //ofStreamMasterOutput << "Market # Placeholder"  << "\n";
        ofStreamMasterOutput << "\n";
    }

    ofStreamMasterOutput.close();

    return 0;
}

int MasterHistory::generate_market_overlap_file() {
    cout << "Generating market overlap file." << endl;

    std::ofstream ofStreamMarketOverlap;
    string strPath = this->strMasterHistoryOutputPath + "/MarketOverlap.csv";
    ofStreamMarketOverlap.open(strPath);

    // Create header row
    ofStreamMarketOverlap << "" << ",";
    ofStreamMarketOverlap << "Sim" << ",";
    ofStreamMarketOverlap << "Market A" << ",";
    ofStreamMarketOverlap << "Market B" << ",";
    ofStreamMarketOverlap << "Num Common Capabilities" << ",";
    ofStreamMarketOverlap << "Percentage Cost Overlap (A^B/A)" << ",";
    ofStreamMarketOverlap << "\n";

    int iRow = 0;
    // Iterate through each simulation
    for (int i = 0; i < this->vecSimulationHistoryPtrs.size(); i++) {
        auto pSimulationHistory = vecSimulationHistoryPtrs.at(i);
        // Iterate through each market A
        for (int j = 0; j < this->iNumMarkets; j++) {
            auto vecMarketOverlap = pSimulationHistory->vecOfVecMarketOverlapMatrix.at(j);
            // Iterate through each market B
            for (int k = 0; k < this->iNumMarkets; k++) {
                double dbPercentOverlap = vecMarketOverlap.at(k);
                int iCommonCapabilities = static_cast<int>(round(this->iCapabilitiesPerMarket * dbPercentOverlap));

                // Insert a row of data
                ofStreamMarketOverlap << iRow << ",";
                ofStreamMarketOverlap << i << ",";
                ofStreamMarketOverlap << j << ",";
                ofStreamMarketOverlap << k << ",";
                ofStreamMarketOverlap << iCommonCapabilities << ",";
                ofStreamMarketOverlap << dbPercentOverlap << ",";
                ofStreamMarketOverlap << "\n";

                // Increment the row counter
                iRow++;
            }
        }
    }

    ofStreamMarketOverlap.close();

    return 0;
}

void MasterHistory::prepare_data_for_output() {
    // ORDER OF OUTPUT FILE PRECEDENCE:
    // 1. Simulation
    // 2. Firm
    // 3. Market
    // 4. Time step
    // (This precedence order means that the rows in the output file are organized such that the info for each time step
    // for a given simulation-firm-market combo is given before moving to the next market. The info for each market for
    // a given simulation-firm combo is given before moving on to the next firm. And the info for each firm is given
    // before moving to the next simulation.)

    // Instantiate all the data rows
    int iNumRows = vecSimulationHistoryPtrs.size() * iNumFirms * iNumMarkets * iMicroStepsPerSim;
    vecDataRows.resize(iNumRows);

    // TODO: factor this out into a separate function
    // Fill in the simulation number, firm ID, agent type, market ID, and time step
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {
                for (int iMicroTimeStep = 0; iMicroTimeStep < iMicroStepsPerSim; iMicroTimeStep++) {
                    int iRow = get_row_number(iSim, iFirm, iMarket, iMicroTimeStep);
                    vecDataRows.at(iRow).iSim = iSim;
                    vecDataRows.at(iRow).iFirmID = iFirm;
                    vecDataRows.at(iRow).strAgentType = vecSimulationHistoryPtrs.at(iSim)->mapFirmToAgentDescription[iFirm];
                    vecDataRows.at(iRow).iMarketID = iMarket;
                    vecDataRows.at(iRow).iMicroTimeStep = iMicroTimeStep;
                }
            }
        }
    }

    fill_in_capital_info();
    fill_in_revenue_info();
    fill_in_market_presence_info();
    fill_in_variable_cost_info(); // Note that variable costs MUST be filled in after market presence has been filled in
    fill_in_fixed_cost_info();
    fill_in_entry_cost_info();
    fill_in_price_info();
    fill_in_quantity_info();
}

void MasterHistory::fill_in_capital_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            // Get all the capital changes corresponding to the simulation-firm combo
            vector<CapitalChange> vecCapitalChanges;
            for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecCapitalChanges) {
                if (entry.iFirmID == iFirm) {
                    vecCapitalChanges.push_back(entry);
                }
            }

            // Get the stating capital amount
            double dbStartingCapital = vecSimulationHistoryPtrs.at(iSim)->mapFirmStartingCapital[iFirm];

            // Account for the case when capital never changed
            if (vecCapitalChanges.empty()) {
                int iStartRow = get_row_number(iSim, iFirm, 0, 0);
                for (int i = 0; i < iNumMarkets * iMicroStepsPerSim; i++) {
                    vecDataRows.at(iStartRow + i).dbCapital = dbStartingCapital;
                }
            }

            int iCurrentTimeStep = 0;
            double dbCurrentCapital = dbStartingCapital;

            // For each entry in the vector of cap changes
            for (int i = 0; i < vecCapitalChanges.size(); i++) {
                auto entry = vecCapitalChanges[i];
                for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {
                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);
                    // Update the rows with the current capital amount
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbCapital = dbCurrentCapital;
                    }

                    // Fill in capital amounts for rows after the last capital change
                    if (i == vecCapitalChanges.size() - 1) {
                        // Update the time step and capital
                        iStartRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current capital amount
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbCapital = entry.dbNewCapitalQty;
                        }
                    }
                } // End of loop through markets

                // Update the time step and capital
                iCurrentTimeStep = entry.iMicroTimeStep;
                dbCurrentCapital = entry.dbNewCapitalQty;

            } // End of loop through vector of capital changes for current simulation-firm combo
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_capital_info()

void MasterHistory::fill_in_revenue_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<RevenueChange> vecRevenueChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecRevenueChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecRevenueChanges.push_back(entry);
                    }
                }

                // NOTE: This starting values will need to be determined dynamically if the simulator is ever
                // modified to allow firms to start with markets in their portfolio.
                double dbStartingRevenue = 0.0;

                // Account for the case when no changes occurred
                if (vecRevenueChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).dbRevenue = dbStartingRevenue;
                    }
                }

                int iCurrentTimeStep = 0;
                double dbCurrentRevenue = dbStartingRevenue;

                // For each entry in the vector of cap changes
                for (int i = 0; i < vecRevenueChanges.size(); i++) {
                    auto entry = vecRevenueChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current capital amount
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbRevenue = dbCurrentRevenue;
                    }

                    // Update the time step and capital
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    dbCurrentRevenue = entry.dbNewRevenueAmount;

                    // Fill in capital amounts for rows after the last revenue change
                    if (i == vecRevenueChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current capital amount
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbRevenue = dbCurrentRevenue;
                        }
                    }
                } // End of loop through vector of revenue changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_revenue_info()


void MasterHistory::fill_in_market_presence_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<MarketPresenceChange> vecMarketPresenceChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecMarketPresenceChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecMarketPresenceChanges.push_back(entry);
                    }
                }

                // NOTE: This starting value will need to be determined dynamically if the simulator is ever
                // modified to allow firms to start with markets in their portfolio.
                bool bStartingPresence = false;

                // Account for the case when no changes occurred
                if (vecMarketPresenceChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).bInMarket = bStartingPresence;
                    }
                }

                int iCurrentTimeStep = 0;
                bool bCurrentPresence = bStartingPresence;

                // For each entry in the vector of market presence changes
                for (int i = 0; i < vecMarketPresenceChanges.size(); i++) {
                    auto entry = vecMarketPresenceChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current market presence value
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).bInMarket = bCurrentPresence;
                    }

                    // Update the time step and market presence value
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    bCurrentPresence = entry.bPresent;

                    // Fill in market presence values for rows after the last market presence change
                    if (i == vecMarketPresenceChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current market presence value
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).bInMarket = bCurrentPresence;
                        }
                    }
                } // End of loop through vector of market presence changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_market_presence_info()

void MasterHistory::fill_in_variable_cost_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        auto mapFirmMarketComboToVarCost = vecSimulationHistoryPtrs.at(iSim)->mapFirmMarketComboToVarCost;
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {
                auto pairFirmMarket = std::make_pair(iFirm, iMarket);
                for (int iMicroStep = 0; iMicroStep < iMicroStepsPerSim; iMicroStep++) {
                    int iRow = get_row_number(iSim, iFirm, iMarket, iMicroStep);
                    if (vecDataRows.at(iRow).bInMarket) {
                        vecDataRows.at(iRow).dbVarCost = mapFirmMarketComboToVarCost[pairFirmMarket];
                    }
                    else {
                        vecDataRows.at(iRow).dbVarCost = 0.0;
                    }
                } // End of loop over micro time steps
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_market_presence_info()


void MasterHistory::fill_in_fixed_cost_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<FixedCostChange> vecFixedCostChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecFixedCostChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecFixedCostChanges.push_back(entry);
                    }
                }

                // NOTE: This starting value will need to be determined dynamically if the simulator is ever
                // modified to allow firms to start with markets in their portfolio.
                double dbStartingFixedCost = 0.0;

                // Account for the case when no changes occurred
                if (vecFixedCostChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).dbFixedCost = dbStartingFixedCost;
                    }
                }

                int iCurrentTimeStep = 0;
                double dbCurrentFixedCost = dbStartingFixedCost;

                // For each entry in the vector of fixed cost changes
                for (int i = 0; i < vecFixedCostChanges.size(); i++) {
                    auto entry = vecFixedCostChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current fixed cost value
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbFixedCost = dbCurrentFixedCost;
                    }

                    // Update the time step and fixed cost
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    dbCurrentFixedCost = entry.dbNewFixedCost;

                    // Fill in fixed cost values for rows after the last fixed cost change
                    if (i == vecFixedCostChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current fixed cost value
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbFixedCost = dbCurrentFixedCost;
                        }
                    }
                } // End of loop through vector of fixed cost changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_fixed_cost_info()


void MasterHistory::fill_in_entry_cost_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        auto mapMaximumEntryCosts = vecSimulationHistoryPtrs.at(iSim)->mapMarketMaximumEntryCost;
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<EntryCostChange> vecEntryCostChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecEntryCostChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecEntryCostChanges.push_back(entry);
                    }
                }

                // Get the starting entry cost for each sim-firm-market combination
                // Note that this will have to be changed to not just be the maximum entry cost for the market if the
                // simulator is modified to allow firms to begin with markets in their portfolios.
                double dbStartingEntryCost = mapMaximumEntryCosts[iMarket];

                // Account for the case when no changes occurred
                if (vecEntryCostChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).dbEntryCost = dbStartingEntryCost;
                    }
                }

                int iCurrentTimeStep = 0;
                double dbCurrentEntryCost = dbStartingEntryCost;

                // For each entry in the vector of entry cost changes
                for (int i = 0; i < vecEntryCostChanges.size(); i++) {
                    auto entry = vecEntryCostChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current entry cost value
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbEntryCost = dbCurrentEntryCost;
                    }

                    // Update the time step and entry cost
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    dbCurrentEntryCost = entry.dbNewEntryCost;

                    // Fill in fixed cost values for rows after the last fixed cost change
                    if (i == vecEntryCostChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current entry cost value
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbEntryCost = dbCurrentEntryCost;
                        }
                    }
                } // End of loop through vector of entry cost changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_entry_cost_info()


void MasterHistory::fill_in_price_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<PriceChange> vecPriceChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecPriceChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecPriceChanges.push_back(entry);
                    }
                }

                // Get the starting price for each sim-firm-market combination
                // Note: This will have to be made dynamic rather than hard-coded later on if the simulator is modified
                // such that firms can start with markets in their portfolios.
                double dbStartingPrice = 0.0;

                // Account for the case when no changes occurred
                if (vecPriceChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).dbPrice = dbStartingPrice;
                    }
                }

                int iCurrentTimeStep = 0;
                double dbCurrentPrice = dbStartingPrice;

                // For each entry in the vector of price changes
                for (int i = 0; i < vecPriceChanges.size(); i++) {
                    auto entry = vecPriceChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current price
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbPrice = dbCurrentPrice;
                    }

                    // Update the time step and price
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    dbCurrentPrice = entry.dbNewPrice;

                    // Fill in fixed cost values for rows after the last price change
                    if (i == vecPriceChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current price
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbPrice = dbCurrentPrice;
                        }
                    }
                } // End of loop through vector of price changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_entry_price_info()


void MasterHistory::fill_in_quantity_info() {
    for (int iSim = 0; iSim < vecSimulationHistoryPtrs.size(); iSim++) {
        for (int iFirm = 0; iFirm < iNumFirms; iFirm++) {
            for (int iMarket = 0; iMarket < iNumMarkets; iMarket++) {

                // Get all the changes corresponding to the simulation-firm-market combo
                vector<ProductionQuantityChange> vecProductionQuantityChanges;

                for (auto entry : vecSimulationHistoryPtrs.at(iSim)->vecProductionQtyChanges) {
                    if (entry.iFirmID == iFirm && entry.iMarketID == iMarket) {
                        vecProductionQuantityChanges.push_back(entry);
                    }
                }

                // Get the starting production quantity for each sim-firm-market combination
                // Note: This will have to be made dynamic rather than hard-coded later on if the simulator is modified
                // such that firms can start with markets in their portfolios.
                double dbStartingQuantity = 0.0;

                // Account for the case when no changes occurred
                if (vecProductionQuantityChanges.empty()) {
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, 0);
                    for (int i = 0; i < iMicroStepsPerSim; i++) {
                        vecDataRows.at(iStartRow + i).dbQty = dbStartingQuantity;
                    }
                }

                int iCurrentTimeStep = 0;
                double dbCurrentQuantity = dbStartingQuantity;

                // For each entry in the vector of price changes
                for (int i = 0; i < vecProductionQuantityChanges.size(); i++) {
                    auto entry = vecProductionQuantityChanges[i];

                    // Get the indices of the rows we want to update
                    int iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                    int iEndRow = get_row_number(iSim, iFirm, iMarket, entry.iMicroTimeStep);

                    // Update the rows with the current production quantity
                    for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                        vecDataRows.at(iRow).dbQty = dbCurrentQuantity;
                    }

                    // Update the time step and production quantity
                    iCurrentTimeStep = entry.iMicroTimeStep;
                    dbCurrentQuantity = entry.dbNewProductionQty;

                    // Fill in fixed cost values for rows after the last production quantity change
                    if (i == vecProductionQuantityChanges.size() - 1) {
                        iStartRow = get_row_number(iSim, iFirm, iMarket, iCurrentTimeStep);
                        iEndRow = get_row_number(iSim, iFirm, iMarket + 1, 0);
                        // Update the rows with the current production quantity
                        for (int iRow = iStartRow; iRow < iEndRow; iRow++) {
                            vecDataRows.at(iRow).dbQty = dbCurrentQuantity;
                        }
                    }
                } // End of loop through vector of production quantity changes for current simulation-firm-market combo
            } // End of loop over markets
        } // End of loop over firms
    } // End of loop over simulations
} // End of fill_in_entry_quantity_info()


int MasterHistory::get_row_number(int iCurrentSim, int iCurrentFirm, int iCurrentMarket, int iCurrentMicroStep) {
    int iRowsFromPastSimulations = iCurrentSim * iNumFirms * iNumMarkets * iMicroStepsPerSim;
    int iRowsFromPastFirms = iCurrentFirm * iNumMarkets * iMicroStepsPerSim;
    int iRowsFromPastMarkets = iCurrentMarket * iMicroStepsPerSim;
    return iRowsFromPastSimulations + iRowsFromPastFirms + iRowsFromPastMarkets + iCurrentMicroStep;
}