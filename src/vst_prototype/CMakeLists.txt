cmake_minimum_required(VERSION 3.22)

project(AdaptiveAutoTune VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE: use a local checkout if given, otherwise download a pinned release.
#   cmake -B build -DJUCE_DIR=/path/to/JUCE     (or set the JUCE_DIR environment variable)
set(JUCE_DIR "$ENV{JUCE_DIR}" CACHE PATH "Path to a local JUCE checkout (empty = download)")
if (JUCE_DIR)
    if (NOT EXISTS "${JUCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "JUCE_DIR=${JUCE_DIR} does not contain a JUCE checkout.")
    endif()
    add_subdirectory(${JUCE_DIR} JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.4
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)
endif()

juce_add_plugin(AdaptiveAutoTune
    COMPANY_NAME "Antigravity"
    PLUGIN_MANUFACTURER_CODE Anti
    PLUGIN_CODE AdAt
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "AdaptiveAutoTune"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
)

# Source files
target_sources(AdaptiveAutoTune PRIVATE
    Source/PluginProcessor.h
    Source/PluginProcessor.cpp
    Source/PluginEditor.h
    Source/PluginEditor.cpp
    Source/Engine.h
    Source/Engine.cpp
    Source/DSP.h
    Source/DSP.cpp
)

# Compile options & definitions
target_compile_definitions(AdaptiveAutoTune PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

# Enable C++20 features
target_compile_features(AdaptiveAutoTune PRIVATE cxx_std_20)

# Link JUCE Modules
target_link_libraries(AdaptiveAutoTune PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_gui_basics
    juce::juce_graphics
    juce::juce_events
    juce::juce_core
)

# Generate a JUCE Header containing include statements for all modules linked
juce_generate_juce_header(AdaptiveAutoTune)
