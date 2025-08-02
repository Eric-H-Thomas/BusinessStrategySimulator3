//
// Created by Eric Thomas on 9/14/23.
//

#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include "../Market/Market.h"
using std::set;
using std::vector;

class MiscUtils {
public:
    static int choose_index_given_probabilities(const vector<double>& probabilities);
    static double dot_product(const vector<double>& vector1, const vector<double>& vector2);
    static int dot_product(const vector<int>& vector1, const vector<int>& vector2);
    static double dot_product(const vector<int>& vector1, const vector<double>& vector2);
    static double dot_product(const vector<double>& vector1, const vector<int>& vector2);
    template<typename T> static T choose_random_from_set(const std::set<T>& inputSet);
    static double get_percentage_overlap(const vector<int>& vector1, const vector<int>& vector2);
    static vector<int> element_wise_logical_and(const vector<int>& vector1, const vector<int>& vector2);
    static vector<int> element_wise_logical_or(const vector<int>& vector1, const vector<int>& vector2);
    static vector<int> vector_addition(const vector<int>& vector1, const vector<int>& vector2);
    static vector<int> vector_subtraction(const vector<int>& vector1, const vector<int>& vector2);
    static void set_all_positive_values_to_one(vector<int>& vector);
    static vector<double> flatten(const vector<vector<double>>& vectorOfVectors);
};