cmake_minimum_required(VERSION 3.16)
project(Chroma CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# mac compatability trick for now
if(APPLE)
    # Create a dummy .cpp so the library can "build"
    set(DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
    file(WRITE ${DUMMY_SRC} "// Empty Chroma implementation for Mac\n")
    set(CHROMA_SOURCES ${DUMMY_SRC})

    # Create a dummy HEADER to replace crm_mth.h
    set(DUMMY_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/mac_include")
    file(MAKE_DIRECTORY ${DUMMY_INC_DIR})
    file(WRITE "${DUMMY_INC_DIR}/crm_mth.h" "#pragma once\n// This file is empty on Mac to prevent build errors.\n")

    # the dummy include directory INSTEAD of the real one
    set(CHROMA_INCLUDE_DIR ${DUMMY_INC_DIR})

    # a macro so C++ code knows Chroma is disabled
    add_compile_definitions(CN_NO_CHROMA)

else()
    # Windows/Linux: Use the Real Files
    set(CHROMA_SOURCES "crm_mth.h" "crm_mth.cpp")
    set(CHROMA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

#  Target
add_library(Chroma STATIC ${CHROMA_SOURCES})

# points to the empty header. On Windows, the real one.
target_include_directories(Chroma PUBLIC ${CHROMA_INCLUDE_DIR})

# Compiler Flags (Windows/Linux Only)
if(MSVC)
    target_compile_options(Chroma PRIVATE "/arch:AVX")
    set_property(TARGET Chroma PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(NOT APPLE)
    target_compile_options(Chroma PRIVATE "-msse" "-mavx")
endif()