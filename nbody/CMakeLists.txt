cmake_minimum_required(VERSION 3.10)
project(nbody)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_COMPILER  mpicc)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fopenmp")
include_directories(include)
include_directories(include/ppp)
include_directories(include/ppp_pnm)

add_executable(nbody
        include/ppp/ppp.h
        include/ppp_pnm/ppp_pnm.h
        src/nbody/CMakeLists.txt
        src/nbody/main.c
        src/nbody/parallel.c
        src/nbody/single.c
        src/ppp_pnm/CMakeLists.txt
        src/ppp_pnm/ppp_pnm.c
        src/CMakeLists.txt
        CMakeLists.txt)

target_link_libraries(nbody
        m
        )