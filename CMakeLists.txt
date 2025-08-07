cmake_minimum_required(VERSION 3.31)
project(BusinessStrategySimulator3)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/pybind11)
include_directories(external/pybind11/include)

# List all the source files used by both the executable and the Pybind11 module
set(COMMON_SOURCES
        WorkingFiles/Config/ConfigValidator.h
        WorkingFiles/Config/ConfigValidator.cpp
        WorkingFiles/Simulator/Simulator.cpp
        WorkingFiles/Simulator/Simulator.h
        WorkingFiles/Agent/ControlAgent.cpp
        WorkingFiles/Agent/ControlAgent.h
        WorkingFiles/Economy/Economy.cpp
        WorkingFiles/Economy/Economy.h
        JSONReader/json.h
        WorkingFiles/Firm/Firm.cpp
        WorkingFiles/Firm/Firm.h
        WorkingFiles/Market/Market.cpp
        WorkingFiles/Market/Market.h
        WorkingFiles/History/SimulationHistory.cpp
        WorkingFiles/History/SimulationHistory.h
        WorkingFiles/History/MasterHistory.cpp
        WorkingFiles/History/MasterHistory.h
        WorkingFiles/Utils/MiscUtils.cpp
        WorkingFiles/Utils/MiscUtils.h
        WorkingFiles/Action/Action.cpp
        WorkingFiles/Action/Action.h
        WorkingFiles/Utils/StringUtils.cpp
        WorkingFiles/Utils/StringUtils.h
        WorkingFiles/DataCache/DataCache.cpp
        WorkingFiles/DataCache/DataCache.h
        WorkingFiles/Agent/BaseAgent.cpp WorkingFiles/Agent/BaseAgent.h WorkingFiles/Agent/StableBaselines3Agent.cpp WorkingFiles/Agent/StableBaselines3Agent.h
        WorkingFiles/PythonAPI/PythonAPI.cpp
        WorkingFiles/PythonAPI/PythonAPI.h
        WorkingFiles/python_wrapper.cpp)


# Executable target
add_executable(BusinessStrategySimulator3 WorkingFiles/main.cpp ${COMMON_SOURCES})
target_link_libraries(BusinessStrategySimulator3 PRIVATE pybind11::embed)

# Pybind11 module target
pybind11_add_module(simulator_module ${COMMON_SOURCES})