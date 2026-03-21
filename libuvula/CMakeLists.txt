cmake_minimum_required(VERSION 3.23)
project(uvula)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(spdlog REQUIRED)
find_package(range-v3 REQUIRED)
find_package(Threads REQUIRED)

option(EXTENSIVE_WARNINGS "Build with all warnings" ON)

option(WITH_PYTHON_BINDINGS "Build with Python bindings: `pyUvula`" ON)
if (WITH_PYTHON_BINDINGS)
    set(PYUVULA_VERSION "1.0.0" CACHE STRING "Version of the pyuvula python bindings")
    message(STATUS "Configuring pyUvula version: ${PYUVULA_VERSION}")
endif ()

add_subdirectory(${PROJECT_SOURCE_DIR}/libuvula)
