#pragma once

#include <JuceHeader.h>
#include "Engine.h"
#include "DSP.h"
#include <atomic>

class AdaptiveAutoTuneAudioProcessor  : public juce::AudioProcessor
{
public:
    AdaptiveAutoTuneAudioProcessor();
    ~AdaptiveAutoTuneAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Parameter layout
    juce::AudioProcessorValueTreeState apvts;

    // Thread-safe values for visualizer polling
    std::atomic<float> z1 { 0.0f };
    std::atomic<float> z2 { 0.0f };
    std::atomic<float> safetySTFT { 1.0f };
    std::atomic<float> safetyACF { 1.0f };
    std::atomic<float> safetyCEP { 1.0f };
    std::atomic<int> regionIndex { 0 }; // 0: transition, 1: periodic, 2: noise, 3: transient, 4: smooth_lp

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // C++ Engine and DSP objects (mono processing, using channel 0 for analysis)
    RepresentationIntelligenceEngine engine;
    AutocorrelationPitchTracker pitchTracker;
    std::vector<DelayPitchShifter> pitchShifters;

    // Circular input buffer for 2048-sample analysis
    std::vector<float> inputCircularBuffer;
    int circularBufferWritePos { 0 };
    int analysisFrameSize { 2048 };

    // DSP state tracking variables
    float lastTargetPitch { 0.0f };
    float smoothedCorrectionSemitones { 0.0f };
    
    // Hop counter to trigger engine analysis (e.g. every 256 samples)
    int samplesSinceLastAnalysis { 0 };
    int analysisHopSize { 256 };

    // Framework parameters adapted in audio thread
    int activeWindowSize { 2048 };
    float activeTroughThreshold { 0.15f };
    bool activeVoicingGate { false };
    bool activeHoldPitch { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveAutoTuneAudioProcessor)
};
