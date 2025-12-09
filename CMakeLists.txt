cmake_minimum_required(VERSION 3.16)
project(Chroma CXX)

# ==========================================
# (C++17)
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
#  Target
# ==========================================
add_library(Chroma STATIC
    "crm_mth.h"
    "crm_mth.cpp"
)

# Allow other projects (like Crimson) to include headers from this folder automatically
target_include_directories(Chroma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ==========================================
# Compiler Flags (SIMD)
# ==========================================
if(MSVC)
    target_compile_options(Chroma PRIVATE "/arch:AVX")
else()
    # GCC / Clang
    target_compile_options(Chroma PRIVATE "-msse" "-mavx")
endif()


# Static Runtime (staticruntime "on")
# This replicates Premake's behavior of forcing /MT (Release) and /MTd (Debug).
# Without this, CMake defaults to /MD and /MDd on Windows.
if(MSVC)
    set_property(TARGET Chroma PROPERTY 
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()