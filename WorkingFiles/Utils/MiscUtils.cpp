//
// Created by Eric Thomas on 9/14/23.
//

#include "MiscUtils.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int MiscUtils::choose_index_given_probabilities(const std::vector<double>& probabilities) {
    // Check if the probabilities vector is empty
    if (probabilities.empty()) {
        std::cerr << "Error: Probabilities vector is empty." << std::endl;
        return -1; // Return an error code
    }


    // Check that the probabilities sum to 100 (probabilities are given as whole percentages)
    double dbSum = 0.0;
    for (double probability : probabilities) {
        dbSum += probability;
    }
    if (std::fabs(100.0 - dbSum) > 1E-12) { // Allow for tiny numerical imprecision
        std::cerr << "Error: Probabilities vector does not sum to 100." << std::endl;
        return -1; // Return an error code
    }


    // Calculate the cumulative probabilities
    std::vector<double> cumulativeProbabilities(probabilities.size());
    cumulativeProbabilities[0] = probabilities[0];
    for (size_t i = 1; i < probabilities.size(); ++i) {
        cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + probabilities[i];
    }

    // Generate a random number between 0 and 100 (probabilities are given as whole percentages)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 100.0);
    double randomValue = distribution(gen);

    // Find the index corresponding to the random value
    for (size_t i = 0; i < cumulativeProbabilities.size(); ++i) {
        if (randomValue < cumulativeProbabilities[i]) {
            return static_cast<int>(i); // Return the chosen index
        }
    }

    // In case something goes wrong (shouldn't happen)
    std::cerr << "Error: Unable to choose an index." << std::endl;
    throw std::exception();
}


// Dot product of two vectors of doubles
[[maybe_unused]] double MiscUtils::dot_product(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    if (vector1.size() != vector2.size()) {
        // Ensure that both vectors have the same dimension.
        std::cerr << "Vectors must have the same dimension" << std::endl;
        throw std::exception();
    }

    double result = 0.0;

    for (size_t i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

// Dot product of two vectors of ints
[[maybe_unused]] int MiscUtils::dot_product(const std::vector<int>& vector1, const std::vector<int>& vector2) {
    if (vector1.size() != vector2.size()) {
        // Ensure that both vectors have the same dimension.
        std::cerr << "Vectors must have the same dimension" << std::endl;
        throw std::exception();
    }

    int result = 0;

    for (size_t i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

// Dot product of a vector of ints with a vector of doubles
[[maybe_unused]] double MiscUtils::dot_product(const std::vector<int>& vector1, const std::vector<double>& vector2) {
    if (vector1.size() != vector2.size()) {
        // Ensure that both vectors have the same dimension.
        std::cerr << "Vectors must have the same dimension" << std::endl;
        throw std::exception();
    }

    double result = 0.0;

    for (size_t i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

// Dot product of a vector of doubles with a vector of ints
[[maybe_unused]] double MiscUtils::dot_product(const std::vector<double>& vector1, const std::vector<int>& vector2) {
    if (vector1.size() != vector2.size()) {
        // Ensure that both vectors have the same dimension.
        std::cerr << "Vectors must have the same dimension" << std::endl;
        throw std::exception();
    }

    double result = 0.0;

    for (size_t i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }

    return result;
}

template <typename T> T MiscUtils::choose_random_from_set(const std::set<T>& inputSet) {
    // Check for an empty set
    if (inputSet.empty()) {
        throw std::invalid_argument("choose_random_from_set: empty set");
    }

    // Convert the set to a vector
    std::vector<T> myVector(inputSet.begin(), inputSet.end());

    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate a random index within the bounds of the vector
    std::uniform_int_distribution<> dis(0, myVector.size() - 1);
    int randomIndex = dis(gen);

    // Return the randomly chosen element
    return myVector[randomIndex];
}

// Explicit template instantiations for choose_random_from_set
template Market MiscUtils::choose_random_from_set(const std::set<Market>& inputSet);
template int MiscUtils::choose_random_from_set(const std::set<int>& inputSet);

double MiscUtils::get_percentage_overlap(const std::vector<int>& vector1, const std::vector<int>& vector2) {
    // Check that the vectors are of the same size and nonempty
    if (vector1.size() != vector2.size() || vector1.empty()) {
        std::cerr << "Error: Attempted to calculate percentage overlap with differently sized and/or empty vectors." << std::endl;
        return -1; // Return an error code
    }

    // Initialize a counter for the overlap
    int overlapCount = 0;

    // Iterate through the vectors and count the overlap
    for (size_t i = 0; i < vector1.size(); ++i) {
        if (vector1[i] == 1 && vector2[i] == 1) {
            overlapCount++;
        }
    }

    // Calculate the percentage overlap
    double percentageOverlap = (static_cast<double>(overlapCount) / static_cast<double>(vector1.size()));

    return percentageOverlap;
}

[[maybe_unused]] vector<int> MiscUtils::element_wise_logical_and(const vector<int>& vector1, const vector<int>& vector2) {
    // Check that the vectors have the same size
    if (vector1.size() != vector2.size()) {
        std::cerr << "Vector sizes must be the same for element-wise AND." << std::endl;
        throw std::exception();
    }

    std::vector<int> result;
    result.reserve(vector1.size()); // Reserve space for the result vector

    // Perform element-wise logical AND and store the result
    for (size_t i = 0; i < vector1.size(); i++) {
        result.push_back(vector1[i] && vector2[i]);
    }

    return result;
}

vector<int> MiscUtils::element_wise_logical_or(const vector<int>& vector1, const vector<int>& vector2) {
    // Check that the vectors have the same size
    if (vector1.size() != vector2.size()) {
        std::cerr << "Vector sizes must be the same for element-wise OR." << std::endl;
        throw std::exception();
    }

    std::vector<int> result;
    result.reserve(vector1.size()); // Reserve space for the result vector

    // Perform element-wise logical OR and store the result
    for (size_t i = 0; i < vector1.size(); i++) {
        result.push_back(vector1[i] || vector2[i]);
    }

    return result;
}

vector<int> MiscUtils::vector_addition(const vector<int>& vector1, const vector<int>& vector2) {
    // Check that the vectors have the same size
    if (vector1.size() != vector2.size()) {
        std::cerr << "Vector sizes must be the same for vector addition." << std::endl;
        throw std::exception();
    }

    std::vector<int> result;
    result.reserve(vector1.size()); // Reserve space for the result vector

    // Perform element-wise addition
    for (size_t i = 0; i < vector1.size(); i++) {
        result.push_back(vector1[i] + vector2[i]);
    }

    return result;
}

vector<int> MiscUtils::vector_subtraction(const vector<int>& vector1, const vector<int>& vector2) {
    // Check that the vectors have the same size
    if (vector1.size() != vector2.size()) {
        std::cerr << "Vector sizes must be the same for vector subtraction." << std::endl;
        throw std::exception();
    }

    std::vector<int> result;
    result.reserve(vector1.size()); // Reserve space for the result vector

    // Perform element-wise addition
    for (size_t i = 0; i < vector1.size(); i++) {
        result.push_back(vector1[i] - vector2[i]);
    }

    return result;
}

void MiscUtils::set_all_positive_values_to_one(vector<int>& vector) {
    for (int& i : vector) {
        if (i > 0) {
            i = 1;
        }
    }
}

vector<double> MiscUtils::flatten(const vector<vector<double>>& vectorOfVectors) {
    vector<double> vecFlattened;
    // Iterate through each vector in the vector of vectors
    for (const auto& innerVector : vectorOfVectors) {
        // Append the elements of the inner vector to the flattened vector
        vecFlattened.insert(vecFlattened.end(), innerVector.begin(), innerVector.end());
    }
    return vecFlattened;
}