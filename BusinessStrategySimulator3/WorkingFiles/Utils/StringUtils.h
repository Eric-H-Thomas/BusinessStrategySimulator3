#pragma once
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

using std::string;

class StringUtils {
public:
    static string toUpper(const string& str);
    static bool equalsIgnoreCase(const string& str1, const string& str2);
    static bool equalsIgnoreCaseAndIgnoreUnderscores(const string& str1, const string& str2);
    static string getTimeStampAsString();
};