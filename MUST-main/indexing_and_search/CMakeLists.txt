cmake_minimum_required(VERSION 3.5)
project(GraphANNS)

set(CMAKE_CXX_STANDARD 17)

if (MODAL2_FLOAT)
    add_compile_definitions(MODAL2_FLOAT)
endif()

if (APPLE)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE OSX_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (OSX_SYSROOT)
        set(CMAKE_OSX_SYSROOT ${OSX_SYSROOT})
        include_directories(SYSTEM "${OSX_SYSROOT}/usr/include/c++/v1")
    endif()
endif()


#### [required] add CGraph dependence ####
include(cmake/GA-CGraph-depend.cmake)

#### [optional] add Eigen3 dependence ####
include(cmake/GA-Eigen3-depend.cmake)

#### [optional] add OpenMP dependence ####
include(cmake/GA-OpenMP-depend.cmake)


set(TUTORIAL_LIST
        T-MSAG
        S-MSAG
        S-MSAB
        S-MABG
        S-MABB
        )

foreach(tut ${TUTORIAL_LIST})
    add_executable(${tut} ${tut}.cpp ${CGRAPH_SRC_LIST})
endforeach()

if (OPENMP_FOUND AND TARGET OpenMP::OpenMP_CXX)
    foreach(tut ${TUTORIAL_LIST})
        target_link_libraries(${tut} PUBLIC OpenMP::OpenMP_CXX)
    endforeach()
endif()


#### [optional] add 3rd-party anns pipeline ####
# add_subdirectory(anns-pipeline/3rd-annoy)
# add_subdirectory(anns-pipeline/3rd-hnsw)
