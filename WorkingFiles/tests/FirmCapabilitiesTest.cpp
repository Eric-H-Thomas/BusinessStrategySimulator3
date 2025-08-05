#include "../Firm/Firm.h"
#include "../Market/Market.h"
#include "../Economy/Economy.h"
#include <vector>
#include <cassert>
#include <iostream>

int main() {
    // Setup economy and markets
    int possibleCaps = 3;
    Firm firm(1, 0.0, possibleCaps);
    Economy economy;

    std::vector<int> caps1{1,0,0};
    std::vector<int> caps2{0,1,0};

    Market market1(1, 0, 0, 0, 0, caps1);
    Market market2(2, 0, 0, 0, 0, caps2);

    economy.add_market(market1);
    economy.add_market(market2);

    // Add market1 to firm's portfolio and capabilities
    firm.add_market_to_portfolio(market1.get_market_id());
    firm.add_market_capabilities_to_firm_capabilities(market1);

    // Attempt to remove capabilities of market2 which is not in portfolio
    int result = firm.remove_market_capabilities_from_firm_capabilities(market2, economy);
    assert(result != 0);

    // Firm's capabilities should remain unchanged
    const std::vector<int>& firmCaps = firm.getVecCapabilities();
    assert(firmCaps == caps1);

    std::cout << "Non-portfolio market removal handled safely" << std::endl;
    return 0;
}
