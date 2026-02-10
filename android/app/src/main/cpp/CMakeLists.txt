cmake_minimum_required(VERSION 3.22.1)
project("nanohindi" C)

# Build the native library
add_library(nanohindi SHARED nanohindi_jni.c)

# Link required libraries
target_link_libraries(nanohindi
    android
    log
    m
)

# Compiler flags for performance
target_compile_options(nanohindi PRIVATE
    -O3
    -ffast-math
    -DNDEBUG
    -D__ANDROID__
)
