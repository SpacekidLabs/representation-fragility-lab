#include "../src/vst_prototype/Source/Engine.h"
#include "../src/vst_prototype/Source/Engine.cpp"
#include "../src/vst_prototype/Source/DSP.h"
#include "../src/vst_prototype/Source/DSP.cpp"
#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    double sampleRate = 44100.0;
    int numSamples = 2048;
    std::vector<float> samples(numSamples, 0.0f);

    // Generate a 440 Hz sine wave
    double freq = 440.0;
    for (int i = 0; i < numSamples; ++i)
    {
        samples[i] = std::sin(2.0 * M_PI * freq * i / sampleRate);
    }

    RepresentationIntelligenceEngine engine;
    FrameworkState state = engine.analyze(samples.data(), numSamples, sampleRate);

    std::cout << "Test 1: 440 Hz Sine wave" << std::endl;
    std::cout << "Coordinates: (" << state.coordinate.first << ", " << state.coordinate.second << ")" << std::endl;
    std::cout << "Region: " << state.region << std::endl;
    std::cout << "ACF Safety: " << state.assumptions["acf"] << std::endl;
    std::cout << "STFT Safety: " << state.assumptions["stft"] << std::endl;
    std::cout << "CEP Safety: " << state.assumptions["cepstrum"] << std::endl;

    // Run pitch tracker
    AutocorrelationPitchTracker pitchTracker;
    float detectedF0 = pitchTracker.detectPitch(samples.data(), numSamples, sampleRate, state.yin_trough);
    std::cout << "Detected Pitch: " << detectedF0 << " Hz" << std::endl;



    // Generate a 150 Hz sine wave
    freq = 150.0;
    for (int i = 0; i < numSamples; ++i)
    {
        samples[i] = std::sin(2.0 * M_PI * freq * i / sampleRate);
    }
    state = engine.analyze(samples.data(), numSamples, sampleRate);

    std::cout << "\nTest 2: 150 Hz Sine wave" << std::endl;
    std::cout << "Coordinates: (" << state.coordinate.first << ", " << state.coordinate.second << ")" << std::endl;
    std::cout << "Region: " << state.region << std::endl;
    std::cout << "ACF Safety: " << state.assumptions["acf"] << std::endl;
    std::cout << "STFT Safety: " << state.assumptions["stft"] << std::endl;
    std::cout << "CEP Safety: " << state.assumptions["cepstrum"] << std::endl;

    detectedF0 = pitchTracker.detectPitch(samples.data(), numSamples, sampleRate, state.yin_trough);
    std::cout << "Detected Pitch: " << detectedF0 << " Hz" << std::endl;

    return 0;
}
