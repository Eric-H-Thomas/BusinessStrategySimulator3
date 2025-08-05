#include "StringUtils.h"
#include <algorithm>
#include <cctype>

string StringUtils::toUpper(const string& str) {
    string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

bool StringUtils::equalsIgnoreCase(const string& str1, const string& str2) {
    if (str1.size() != str2.size()) {
        return false;
    }
    for (size_t i = 0; i < str1.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(str1[i])) !=
            std::tolower(static_cast<unsigned char>(str2[i]))) {
            return false;
        }
    }
    return true;
}

bool StringUtils::equalsIgnoreCaseAndIgnoreUnderscores(const string& str1, const string& str2) {
    // Remove underscores from both strings and then compare them case-insensitively
    string str1NoUnderscores = str1;
    str1NoUnderscores.erase(std::remove(str1NoUnderscores.begin(), str1NoUnderscores.end(), '_'), str1NoUnderscores.end());

    string str2NoUnderscores = str2;
    str2NoUnderscores.erase(std::remove(str2NoUnderscores.begin(), str2NoUnderscores.end(), '_'), str2NoUnderscores.end());

    return equalsIgnoreCase(str1NoUnderscores, str2NoUnderscores);
}