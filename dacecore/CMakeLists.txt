# This CMakeLists.txt will build only the DACE Core.

# This is a reduced version of the original file
# (https://github.com/dacelib/dace/blob/master/CMakeLists.txt)

cmake_minimum_required (VERSION 3.0.0)

project(DACECORE C)

set(DACE_MAJOR_VERSION 2)
set(DACE_MINOR_VERSION 0)
set(DACE_PATCH_VERSION 1)

# since MacOS 10.14 (XCode 10.0), default includes are no longer installed to /usr/include or /usr/local/include
# the SDK to use must be specified explicitly in CMAKE_OSX_SYSROOT as it seems at least cmake 3.12.4 does not find it automatically if left empty
# this must happen before project()
if(APPLE AND NOT DEFINED ENV{SDKROOT})
  execute_process(COMMAND xcodebuild -sdk macosx -version Path OUTPUT_VARIABLE RES OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_OSX_SYSROOT "${RES}" CACHE PATH "The product will be built against the headers and libraries located inside the indicated SDK.")
endif(APPLE AND NOT DEFINED ENV{SDKROOT})

# add our own CMake modules to path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/dacelib/cmake)

# global OS specific settings
if(WIN32)
  cmake_policy(SET CMP0054 NEW)
elseif(APPLE)
  cmake_policy(SET CMP0042 NEW)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_NAME_DIR "@loader_path")
endif(WIN32)

# Set C standard (adapted https://github.com/ceres-solver/ceres-solver/blob/master/CMakeLists.txt)
# Respect user-specified CMAKE_C or default to C99.
# Standard is a hard requirement and that option is hidden from CMake GUI
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99 CACHE STRING "C standard (minimum 99)" FORCE)
  set_property(CACHE CMAKE_C_STANDARD PROPERTY STRINGS 99 11)  # options shown in CMake GUI
endif(NOT CMAKE_C_STANDARD)
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "")
mark_as_advanced(CMAKE_C_STANDARD_REQUIRED)

# Set generally understood compiler flags (enforcing -O2)
add_compile_options("$<$<CONFIG:Release>:-O2>" "$<$<CONFIG:RelWithDebInfo>:-O2>")

# Set compiler specific flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  add_compile_options("-Wall" "$<$<CONFIG:Release>:-frounding-math>" "$<$<CONFIG:Release>:-fsignaling-nans>" "$<$<CONFIG:RelWithDebInfo>:-frounding-math>" "$<$<CONFIG:RelWithDebInfo>:-fsignaling-nans>")
elseif(CMAKE_C_COMPILER_ID MATCHES ".*Clang")        # Also catch IDs like "AppleClang"
  add_compile_options("-Wall")
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  add_compile_options("/W4" "$<$<CONFIG:Release>:/fp:strict>" "$<$<CONFIG:RelWithDebInfo>:/fp:strict>")
endif(CMAKE_C_COMPILER_ID STREQUAL "GNU")

# Tri-state option for memory model and the associated choices
set(DACE_MEMORY_MODEL DYNAMIC)
string(TOUPPER "${DACE_MEMORY_MODEL}" DACE_MEMORY_MODEL_U)

# Default build type option if not set already
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
       "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
       FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)  # options shown in CMake GUI
endif(NOT CMAKE_BUILD_TYPE)

# Debug flag
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_U)
if(CMAKE_BUILD_TYPE_U STREQUAL "DEBUG")
  set(WITH_DEBUG 1)
endif(CMAKE_BUILD_TYPE_U STREQUAL "DEBUG")

# Run checks for platform-specific functions
include(CheckSafeStrings)
check_safe_strings(HAVE_SAFE_STRINGS)

# Pass options to source by configuring config.h file in core
configure_file(dacelib/core/include/dace/config.h.in dacelib/core/include/dace/config.h)

# Globally set the include directory for the C core headers
include_directories(dacelib/core/include)
include_directories(${CMAKE_BINARY_DIR}/dacelib/core/include)
include_directories(dacelib/core/contrib/include)

# Add dacecore library
add_library(
  dacecore STATIC
  dacelib/core/daceaux.c
  dacelib/core/dacebasic.c
  dacelib/core/dacecompat.c
  dacelib/core/daceerror.c
  dacelib/core/daceeval.c
  dacelib/core/daceinit.c
  dacelib/core/daceio.c
  dacelib/core/dacemath.c
  dacelib/core/dacememory.c
  dacelib/core/dacenorm.c
  dacelib/core/contrib/libf2c.c
  dacelib/core/contrib/dgamma.c
  dacelib/core/contrib/ribesl.c
  dacelib/core/contrib/rjbesl.c
  dacelib/core/contrib/rkbesl.c
  dacelib/core/contrib/rybesl.c
  dacelib/core/contrib/psi.c
  dacelib/core/contrib/zeta.c
)

target_compile_definitions(dacecore PRIVATE "DACE_API=")

install(
  TARGETS dacecore
  LIBRARY DESTINATION lib COMPONENT libraries
  PUBLIC_HEADER DESTINATION include COMPONENT headers
)
