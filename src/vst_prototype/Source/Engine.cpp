#include "Engine.h"
#include <cmath>
#include <numeric>
#include <algorithm>

// Define MU, SIGMA, and VT constants matching weights.py
const float RepresentationIntelligenceEngine::MU[10] = {
    0.76195118f, 0.29042404f, 0.11610899f, 0.63087928f, 2.64905438f,
    0.56809152f, 3702.86576f, 0.71430379f, 2.86565869f, 0.05771320f
};

const float RepresentationIntelligenceEngine::SIGMA[10] = {
    0.08931102f, 0.19573561f, 0.16632286f, 0.25154298f, 0.96131249f,
    0.10439500f, 2717.09600f, 0.16113735f, 1.62378944f, 0.05117988f
};

const float RepresentationIntelligenceEngine::VT[2][10] = {
    {0.47315060f, 0.46611747f, 0.32685922f, -0.12470205f, 0.11155174f, -0.01194378f, 0.44399277f, -0.46931129f, 0.04403310f, -0.06758288f},
    {-0.11994855f, -0.16311870f, 0.19125473f, -0.39156500f, 0.52897854f, -0.25676939f, -0.16203983f, -0.06562160f, 0.52857375f, 0.33675064f}
};

const float RepresentationIntelligenceEngine::W_STFT[6] = {
    0.84399633f, -0.02332773f, -0.03847526f, -0.00716794f, -0.00057247f, 0.00734064f
};

const float RepresentationIntelligenceEngine::W_ACF[6] = {
    0.86417094f, -0.00584328f, -0.04004930f, -0.00749728f, -0.00053847f, 0.00706432f
};

const float RepresentationIntelligenceEngine::W_CEP[6] = {
    0.66975406f, -0.01080766f, -0.02257589f, -0.00841665f, -0.00032442f, 0.00465065f
};

const float RepresentationIntelligenceEngine::W_CQT[6] = {
    0.85911548f, -0.01305596f, -0.03705924f, -0.00408812f, -0.00047205f, 0.00600636f
};

const float RepresentationIntelligenceEngine::W_CWT[6] = {
    0.88822169f, 0.00137673f, -0.03561487f, -0.00836219f, -0.00049869f, 0.00658515f
};

RepresentationIntelligenceEngine::RepresentationIntelligenceEngine()
{
    fft1024 = std::make_unique<juce::dsp::FFT>(10); // 1024 points
    fft2048 = std::make_unique<juce::dsp::FFT>(11); // 2048 points
    fft4096 = std::make_unique<juce::dsp::FFT>(12); // 4096 points
}

FrameworkState RepresentationIntelligenceEngine::analyze(const float* frameSamples, int numSamples, double sampleRate)
{
    // Compute RMS and basic stats
    float sumSq = 1e-12f;
    float maxVal = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        float absVal = std::abs(frameSamples[i]);
        sumSq += frameSamples[i] * frameSamples[i];
        if (absVal > maxVal) maxVal = absVal;
    }
    float rms = std::sqrt(sumSq / numSamples);

    // 1. ZCR
    float zcr = computeZCR(frameSamples, numSamples);

    // 2. Crest Factor
    float crestFactor = computeCrestFactor(frameSamples, numSamples, rms);

    // 3. Kurtosis
    float kurtosis = computeKurtosis(frameSamples, numSamples, rms);

    // 4. ACF-derived Periodicity and Harmonic Ratio
    std::vector<float> acf = computeACF(frameSamples, numSamples);
    std::pair<float, float> acfFeats = computeACFFeatures(acf, sampleRate);
    float harmonicRatio = acfFeats.first;
    float periodicity = acfFeats.second;

    // 5. Spectral features from FFT Magnitude
    std::vector<float> mag = computeMagnitudeSpectrum(frameSamples, numSamples);
    float entropy = computeSpectralEntropy(mag);
    float flatness = computeSpectralFlatness(mag);
    float rolloff = computeSpectralRolloff(mag, sampleRate);
    float sparsity = computeHoyerSparsity(mag);
    float centroid = computeSpectralCentroid(mag, sampleRate);

    // Package the 10 descriptors
    // Features array order matches MU/SIGMA:
    // [entropy, flatness, zcr, harmonic_ratio, crest_factor, periodicity, rolloff, Hoyer, kurtosis, centroid]
    float features[10];
    features[0] = entropy;
    features[1] = flatness;
    features[2] = zcr;
    features[3] = harmonicRatio;
    features[4] = crestFactor;
    features[5] = periodicity;
    features[6] = rolloff;
    features[7] = sparsity;
    features[8] = kurtosis;
    features[9] = centroid;

    // Standardize features
    float featuresStd[10];
    for (int i = 0; i < 10; ++i) {
        featuresStd[i] = (features[i] - MU[i]) / SIGMA[i];
    }

    // PCA Projection to 2D coordinates (z1, z2)
    float z1 = 0.0f;
    float z2 = 0.0f;
    for (int j = 0; j < 10; ++j) {
        z1 += featuresStd[j] * VT[0][j];
        z2 += featuresStd[j] * VT[1][j];
    }

    // Identify semantic region
    std::string region;
    if (z1 > 1.5f) {
        region = "noise_collapse";
    } else if (z1 < -0.5f && z2 < -0.2f) {
        region = "periodic_harmonic";
    } else if (z2 > 1.5f) {
        region = "transient_overloaded";
    } else if (z1 >= -0.5f && z1 <= 1.5f && z2 < -0.2f) {
        region = "smooth_lowpass";
    } else {
        region = "transition_zone";
    }

    // Evaluate Safety Scores (degree-2 polynomial)
    // poly features: [1, z1, z2, z1^2, z2^2, z1*z2]
    float poly[6] = {
        1.0f,
        z1,
        z2,
        z1 * z1,
        z2 * z2,
        z1 * z2
    };

    auto calcScore = [&](const float* weights) {
        float score = 0.0f;
        for (int i = 0; i < 6; ++i) {
            score += poly[i] * weights[i];
        }
        return std::clamp(score, 0.0f, 1.0f);
    };

    std::map<std::string, float> assumptions;
    assumptions["stft"] = calcScore(W_STFT);
    assumptions["acf"]  = calcScore(W_ACF);
    assumptions["cepstrum"] = calcScore(W_CEP);
    assumptions["cqt"]  = calcScore(W_CQT);
    assumptions["wavelet"] = calcScore(W_CWT);

    // Compute recommendations
    int recommendedWindow = 2048;
    if (region == "noise_collapse") {
        recommendedWindow = 4096;
    } else if (region == "transient_overloaded") {
        recommendedWindow = 1024;
    }

    // Pre-calibrated DSP parameters
    float stftSafety = assumptions["stft"];
    float denoisingAlpha = 0.5f + 3.5f * (1.0f - stftSafety);
    float denoisingBeta = 0.02f;
    if (region == "noise_collapse") {
        denoisingBeta = 0.06f;
    } else if (region == "periodic_harmonic") {
        denoisingBeta = 0.005f;
    }

    float yinTrough = 0.15f;
    if (region == "noise_collapse") {
        yinTrough = 0.25f;
    } else if (region == "transient_overloaded") {
        yinTrough = 0.08f;
    }

    bool voicingGate = (region == "periodic_harmonic" || region == "smooth_lowpass");
    bool holdPitch = (region == "transient_overloaded");
    float onsetThreshold = (region == "noise_collapse") ? 0.30f : 0.15f;

    FrameworkState state;
    state.coordinate = { z1, z2 };
    state.region = region;
    state.assumptions = assumptions;
    state.recommended_window = recommendedWindow;
    state.recommended_latency = recommendedWindow / 2;
    state.denoising_alpha = denoisingAlpha;
    state.denoising_beta = denoisingBeta;
    state.yin_trough = yinTrough;
    state.voicing_gate = voicingGate;
    state.hold_pitch = holdPitch;
    state.onset_threshold = onsetThreshold;

    return state;
}

std::vector<float> RepresentationIntelligenceEngine::computeACF(const float* frame, int numSamples)
{
    std::vector<float> acf(numSamples, 0.0f);
    for (int tau = 0; tau < numSamples; ++tau) {
        float sum = 0.0f;
        for (int t = 0; t < numSamples - tau; ++t) {
            sum += frame[t] * frame[t + tau];
        }
        acf[tau] = sum;
    }
    float r0 = acf[0] + 1e-12f;
    for (int i = 0; i < numSamples; ++i) {
        acf[i] /= r0;
    }
    return acf;
}

std::vector<float> RepresentationIntelligenceEngine::computeMagnitudeSpectrum(const float* frame, int numSamples)
{
    juce::dsp::FFT* activeFFT = nullptr;
    if (numSamples == 1024)      activeFFT = fft1024.get();
    else if (numSamples == 2048) activeFFT = fft2048.get();
    else if (numSamples == 4096) activeFFT = fft4096.get();
    else                         activeFFT = fft2048.get(); // fallback

    // Pad to 2N for real forward transform
    std::vector<float> fftBuffer(numSamples * 2, 0.0f);
    std::copy(frame, frame + numSamples, fftBuffer.begin());

    activeFFT->performRealOnlyForwardTransform(fftBuffer.data());

    int nBins = numSamples / 2 + 1; // e.g. 1025 bins for 2048 samples
    std::vector<float> mag(nBins, 0.0f);
    
    // Bin 0
    mag[0] = std::abs(fftBuffer[0]);
    // Bins 1 to N-1
    for (int k = 1; k < nBins - 1; ++k) {
        float realPart = fftBuffer[2 * k];
        float imagPart = fftBuffer[2 * k + 1];
        mag[k] = std::sqrt(realPart * realPart + imagPart * imagPart);
    }
    // Bin N
    mag[nBins - 1] = std::abs(fftBuffer[numSamples]);

    return mag;
}

float RepresentationIntelligenceEngine::computeSpectralEntropy(const std::vector<float>& mag)
{
    float sum = 0.0f;
    for (float m : mag) sum += m;
    if (sum < 1e-12f) return 1.0f;

    float entropySum = 0.0f;
    for (float m : mag) {
        float p = m / sum;
        if (p > 1e-12f) {
            entropySum += p * std::log2(p);
        }
    }
    float maxEntropy = std::log2((float)mag.size());
    return -entropySum / maxEntropy;
}

float RepresentationIntelligenceEngine::computeSpectralFlatness(const std::vector<float>& mag)
{
    float sum = 0.0f;
    float sumLog = 0.0f;
    for (float m : mag) {
        sum += m;
        sumLog += std::log(m + 1e-12f);
    }
    float arithmeticMean = sum / mag.size();
    float geometricMean = std::exp(sumLog / mag.size());
    return geometricMean / (arithmeticMean + 1e-12f);
}

float RepresentationIntelligenceEngine::computeZCR(const float* frame, int numSamples)
{
    float sumDiff = 0.0f;
    for (int i = 1; i < numSamples; ++i) {
        float sign1 = (frame[i] >= 0.0f) ? 1.0f : -1.0f;
        float sign0 = (frame[i - 1] >= 0.0f) ? 1.0f : -1.0f;
        sumDiff += std::abs(sign1 - sign0);
    }
    return (sumDiff / (numSamples - 1)) / 2.0f;
}

std::pair<float, float> RepresentationIntelligenceEngine::computeACFFeatures(const std::vector<float>& acf, double sampleRate)
{
    int minLag = static_cast<int>(sampleRate / 1000.0); // pitch fmax = 1000Hz
    int maxLag = static_cast<int>(sampleRate / 80.0);   // pitch fmin = 80Hz
    if (maxLag >= (int)acf.size()) maxLag = (int)acf.size() - 1;

    if (minLag >= maxLag) {
        return { 0.0f, 0.0f };
    }

    int peakLag = minLag;
    float maxVal = -1.0f;
    float minVal = 2.0f;
    float acfSum = 0.0f;

    for (int lag = minLag; lag <= maxLag; ++lag) {
        float val = acf[lag];
        acfSum += val;
        if (val > maxVal) {
            maxVal = val;
            peakLag = lag;
        }
        if (val < minVal) {
            minVal = val;
        }
    }

    int numLags = maxLag - minLag + 1;
    float acfMean = acfSum / numLags;

    float harmonicRatio = maxVal; // ACF normalized: acf[0]=1.0
    float periodicity = (maxVal - acfMean) / (maxVal - minVal + 1e-10f);
    periodicity = std::clamp(periodicity, 0.0f, 1.0f);

    return { harmonicRatio, periodicity };
}

float RepresentationIntelligenceEngine::computeCrestFactor(const float* frame, int numSamples, float rms)
{
    float maxVal = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        float absVal = std::abs(frame[i]);
        if (absVal > maxVal) maxVal = absVal;
    }
    return maxVal / (rms + 1e-12f);
}

float RepresentationIntelligenceEngine::computeSpectralRolloff(const std::vector<float>& mag, double sampleRate)
{
    float totalSum = 0.0f;
    for (float m : mag) totalSum += m;
    if (totalSum < 1e-12f) return 0.0f;

    float threshold = 0.85f * totalSum;
    float runningSum = 0.0f;
    int rolloffBin = 0;
    for (int i = 0; i < (int)mag.size(); ++i) {
        runningSum += mag[i];
        if (runningSum >= threshold) {
            rolloffBin = i;
            break;
        }
    }
    float binWidth = static_cast<float>(sampleRate / (2.0 * (mag.size() - 1)));
    return rolloffBin * binWidth;
}

float RepresentationIntelligenceEngine::computeHoyerSparsity(const std::vector<float>& mag)
{
    float l1 = 0.0f;
    float l2Sq = 0.0f;
    for (float m : mag) {
        l1 += std::abs(m);
        l2Sq += m * m;
    }
    float l2 = std::sqrt(l2Sq);
    if (l2 < 1e-12f) return 1.0f;

    float n = static_cast<float>(mag.size());
    float ratio = l1 / l2;
    float sqrtN = std::sqrt(n);
    return (sqrtN - ratio) / (sqrtN - 1.0f);
}

float RepresentationIntelligenceEngine::computeKurtosis(const float* frame, int numSamples, float rms)
{
    float sumVal = 0.0f;
    for (int i = 0; i < numSamples; ++i) sumVal += frame[i];
    float mean = sumVal / numSamples;

    float sumFourthDiff = 0.0f;
    float sumSqDiff = 1e-12f;
    for (int i = 0; i < numSamples; ++i) {
        float diff = frame[i] - mean;
        sumSqDiff += diff * diff;
        sumFourthDiff += diff * diff * diff * diff;
    }

    float var = sumSqDiff / numSamples;
    float fourthMoment = sumFourthDiff / numSamples;
    return fourthMoment / (var * var + 1e-12f);
}

float RepresentationIntelligenceEngine::computeSpectralCentroid(const std::vector<float>& mag, double sampleRate)
{
    float numSum = 0.0f;
    float denSum = 0.0f;
    float binWidth = static_cast<float>(sampleRate / (2.0 * (mag.size() - 1)));

    for (int i = 0; i < (int)mag.size(); ++i) {
        float freq = i * binWidth;
        numSum += mag[i] * freq;
        denSum += mag[i];
    }
    if (denSum < 1e-12f) return 0.0f;
    return numSum / denSum;
}
