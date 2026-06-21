#pragma once

#include <JuceHeader.h>
#include <vector>
#include <string>
#include <map>

struct FrameworkState
{
    std::pair<float, float> coordinate; // (z1, z2) PCA coordinates
    std::string region;                 // Semantic region string
    std::map<std::string, float> assumptions; // maps representation -> safety score [0, 1]
    int recommended_window;
    int recommended_latency;
    
    // Recommended parameters for various tasks
    float denoising_alpha;
    float denoising_beta;
    float yin_trough;
    bool voicing_gate;
    bool hold_pitch;
    float onset_threshold;
};

class RepresentationIntelligenceEngine
{
public:
    RepresentationIntelligenceEngine();
    ~RepresentationIntelligenceEngine() = default;

    FrameworkState analyze (const float* frameSamples, int numSamples, double sampleRate);

private:
    std::vector<float> computeACF (const float* frame, int numSamples);
    std::vector<float> computeMagnitudeSpectrum (const float* frame, int numSamples);
    
    float computeSpectralEntropy (const std::vector<float>& mag);
    float computeSpectralFlatness (const std::vector<float>& mag);
    float computeZCR (const float* frame, int numSamples);
    std::pair<float, float> computeACFFeatures (const std::vector<float>& acf, double sampleRate);
    float computeCrestFactor (const float* frame, int numSamples, float rms);
    float computeSpectralRolloff (const std::vector<float>& mag, double sampleRate);
    float computeHoyerSparsity (const std::vector<float>& mag);
    float computeKurtosis (const float* frame, int numSamples, float rms);
    float computeSpectralCentroid (const std::vector<float>& mag, double sampleRate);

    // Pre-trained SVD / normalization weights from weights.py
    static const float MU[10];
    static const float SIGMA[10];
    static const float VT[2][10];
    
    // Ridge regression safety weights (degree-2 polynomial)
    static const float W_STFT[6];
    static const float W_ACF[6];
    static const float W_CEP[6];
    static const float W_CQT[6];
    static const float W_CWT[6];

    std::unique_ptr<juce::dsp::FFT> fft1024;
    std::unique_ptr<juce::dsp::FFT> fft2048;
    std::unique_ptr<juce::dsp::FFT> fft4096;
};
