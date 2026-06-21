#pragma once

#include <JuceHeader.h>
#include <vector>

class AutocorrelationPitchTracker
{
public:
    AutocorrelationPitchTracker() = default;
    ~AutocorrelationPitchTracker() = default;

    float detectPitch (const float* samples, int numSamples, double sampleRate, float troughThreshold = 0.15f);
};

class DelayPitchShifter
{
public:
    DelayPitchShifter();
    ~DelayPitchShifter() = default;

    void init (double sampleRate);
    void clear();

    // Process a single sample, returning the pitch-shifted output
    float processSample (float inputSample, float pitchShiftSemitones, float currentF0 = 0.0f);

private:
    double sampleRate { 44100.0 };
    
    // Circular delay buffer
    std::vector<float> delayBuffer;
    int bufferSize { 16384 };
    int writeIndex { 0 };

    // Two read taps for cross-fading
    float tap1 { 0.0f };
    float tap2 { 0.0f };
    
    // Fading phase
    float phase { 0.0f };
    float tapOffset { 0.0f };
    
    // Window length in samples for cross-fading
    float windowSize { 2048.0f };
};
