#include "DSP.h"
#include <cmath>
#include <vector>
#include <algorithm>

// -----------------------------------------------------------------------------
// YIN-based Autocorrelation Pitch Tracker
// -----------------------------------------------------------------------------
float AutocorrelationPitchTracker::detectPitch(const float* samples, int numSamples, double sampleRate, float troughThreshold)
{
    int minLag = static_cast<int>(sampleRate / 1000.0); // max 1000 Hz
    int maxLag = static_cast<int>(sampleRate / 80.0);   // min 80 Hz
    if (maxLag >= numSamples / 2) maxLag = numSamples / 2 - 1;

    if (minLag >= maxLag) return 0.0f;

    // 1. Difference function
    std::vector<float> d(maxLag + 1, 0.0f);
    for (int tau = 1; tau <= maxLag; ++tau) {
        float diffSum = 0.0f;
        for (int i = 0; i < numSamples / 2; ++i) {
            float diff = samples[i] - samples[i + tau];
            diffSum += diff * diff;
        }
        d[tau] = diffSum;
    }
    d[0] = 1.0f;

    // 2. Cumulative mean normalized difference function
    std::vector<float> dPrime(maxLag + 1, 0.0f);
    dPrime[0] = 1.0f;
    float runningSum = 0.0f;
    for (int tau = 1; tau <= maxLag; ++tau) {
        runningSum += d[tau];
        if (runningSum > 1e-12f) {
            dPrime[tau] = d[tau] / (runningSum / tau);
        } else {
            dPrime[tau] = 1.0f;
        }
    }

    // 3. Absolute thresholding (first local minimum below threshold)
    int periodLag = -1;
    for (int tau = minLag + 1; tau < maxLag; ++tau) {
        if (dPrime[tau] < troughThreshold) {
            // Check if it is a local minimum (trough)
            if (dPrime[tau] < dPrime[tau - 1] && dPrime[tau] < dPrime[tau + 1]) {
                periodLag = tau;
                break;
            }
        }
    }

    // Fallback to global minimum in range if no local minimum meets threshold
    if (periodLag == -1) {
        float minVal = 100.0f;
        for (int tau = minLag; tau <= maxLag; ++tau) {
            if (dPrime[tau] < minVal) {
                minVal = dPrime[tau];
                periodLag = tau;
            }
        }
        if (minVal > 0.60f) {
            return 0.0f;
        }
    }

    // 4. Parabolic interpolation for sub-sample precision
    float floatLag = (float)periodLag;
    if (periodLag > minLag && periodLag < maxLag) {
        float alpha = dPrime[periodLag - 1];
        float beta = dPrime[periodLag];
        float gamma = dPrime[periodLag + 1];
        float denom = alpha - 2.0f * beta + gamma;
        if (std::abs(denom) > 1e-6f) {
            float offset = 0.5f * (alpha - gamma) / denom;
            if (std::abs(offset) <= 1.0f) {
                floatLag = (float)periodLag + offset;
            }
        }
    }

    if (floatLag > 0.0f) {
        float freq = static_cast<float>(sampleRate) / floatLag;
        if (freq >= 80.0f && freq <= 1000.0f) {
            return freq;
        }
    }

    return 0.0f;
}

// -----------------------------------------------------------------------------
// Real-Time Delay Pitch Shifter
// -----------------------------------------------------------------------------
DelayPitchShifter::DelayPitchShifter()
{
    delayBuffer.resize(bufferSize, 0.0f);
}

void DelayPitchShifter::init(double sr)
{
    sampleRate = sr;
    clear();
}

void DelayPitchShifter::clear()
{
    std::fill(delayBuffer.begin(), delayBuffer.end(), 0.0f);
    writeIndex = 0;
    phase = 0.0f;
}

float DelayPitchShifter::processSample(float inputSample, float pitchShiftSemitones, float currentF0)
{
    // Write input sample
    delayBuffer[writeIndex] = inputSample;

    // Dynamically adjust window size to be an even multiple of the pitch period
    if (currentF0 > 50.0f && currentF0 < 1000.0f)
    {
        float periodSamples = (float)(sampleRate / currentF0);
        // Target a window size around 40 ms (e.g. 1764 samples at 44.1 kHz)
        float targetWindow = 0.04f * (float)sampleRate;
        int K = 2 * (int)std::round (targetWindow / (2.0f * periodSamples));
        if (K < 2) K = 2;
        float targetWindowSamples = (float)K * periodSamples;

        // Smoothly adjust windowSize to targetWindowSamples
        float targetWindowSmooth = windowSize + 0.05f * (targetWindowSamples - windowSize);
        
        // Scale phase to avoid jumps when windowSize changes
        if (std::abs (targetWindowSmooth - windowSize) > 1e-4f)
        {
            phase = phase * (targetWindowSmooth / windowSize);
            windowSize = targetWindowSmooth;
        }
    }
    else
    {
        // Smoothly return to default window size of 2048
        float targetWindowSmooth = windowSize + 0.05f * (2048.0f - windowSize);
        if (std::abs (targetWindowSmooth - windowSize) > 1e-4f)
        {
            phase = phase * (targetWindowSmooth / windowSize);
            windowSize = targetWindowSmooth;
        }
    }

    // Calculate shift ratio and delay rate of change
    float ratio = std::pow(2.0f, pitchShiftSemitones / 12.0f);
    float rate = 1.0f - ratio;

    // Accumulate phase
    phase += rate;
    if (phase >= windowSize)
    {
        phase -= windowSize;
    }
    else if (phase < 0.0f)
    {
        phase += windowSize;
    }

    // Tap offsets (Tap 2 is 180 degrees out of phase with Tap 1)
    float offset1 = phase;
    float offset2 = phase + windowSize / 2.0f;
    if (offset2 >= windowSize)
    {
        offset2 -= windowSize;
    }

    // Safety offset to avoid reading from writeIndex
    float baseDelay = 512.0f;

    // Read Tap 1
    float readPos1 = (float)writeIndex - (offset1 + baseDelay);
    while (readPos1 < 0.0f) readPos1 += (float)bufferSize;
    int idx1 = (int)readPos1;
    float frac1 = readPos1 - idx1;
    int idx1Next = (idx1 + 1) % bufferSize;
    float s1 = delayBuffer[idx1] * (1.0f - frac1) + delayBuffer[idx1Next] * frac1;

    // Read Tap 2
    float readPos2 = (float)writeIndex - (offset2 + baseDelay);
    while (readPos2 < 0.0f) readPos2 += (float)bufferSize;
    int idx2 = (int)readPos2;
    float frac2 = readPos2 - idx2;
    int idx2Next = (idx2 + 1) % bufferSize;
    float s2 = delayBuffer[idx2] * (1.0f - frac2) + delayBuffer[idx2Next] * frac2;

    // Cross-fade window logic: triangular window
    float w1 = 0.0f;
    if (offset1 < windowSize / 2.0f)
    {
        w1 = offset1 / (windowSize / 2.0f);
    }
    else
    {
        w1 = (windowSize - offset1) / (windowSize / 2.0f);
    }
    float w2 = 1.0f - w1;

    // Combine taps
    float outSample = s1 * w1 + s2 * w2;

    // Advance write pointer
    writeIndex = (writeIndex + 1) % bufferSize;

    return outSample;
}
