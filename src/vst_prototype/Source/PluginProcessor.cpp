#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

static float snapToScale (float rawMidiNote, int scaleIndex, int rootIndex)
{
    if (scaleIndex == 0) // Chromatic
        return std::round (rawMidiNote);

    // Scale patterns relative to root
    std::vector<int> pattern;
    switch (scaleIndex)
    {
        case 1: pattern = { 0, 2, 4, 5, 7, 9, 11 }; break; // Major
        case 2: pattern = { 0, 2, 3, 5, 7, 8, 10 }; break; // Natural Minor
        case 3: pattern = { 0, 2, 4, 7, 9 };         break; // Pentatonic Major
        case 4: pattern = { 0, 3, 5, 7, 10 };        break; // Pentatonic Minor
        default: return std::round (rawMidiNote);
    }

    // Transpose pattern by rootIndex modulo 12
    std::vector<int> validClasses;
    for (int pc : pattern)
    {
        validClasses.push_back ((pc + rootIndex) % 12);
    }

    // Find the nearest note in validClasses
    int baseMidi = (int)std::round (rawMidiNote);
    
    int closestMidi = baseMidi;
    float minDist = 999.0f;

    // Search in a range of ±12 semitones around baseMidi
    for (int checkMidi = baseMidi - 12; checkMidi <= baseMidi + 12; ++checkMidi)
    {
        int pc = checkMidi % 12;
        if (pc < 0) pc += 12;

        if (std::find (validClasses.begin(), validClasses.end(), pc) != validClasses.end())
        {
            float dist = std::abs (rawMidiNote - (float)checkMidi);
            if (dist < minDist)
            {
                minDist = dist;
                closestMidi = checkMidi;
            }
        }
    }

    return (float)closestMidi;
}

AdaptiveAutoTuneAudioProcessor::AdaptiveAutoTuneAudioProcessor()
    : AudioProcessor (BusesProperties()
                      .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     ),
      apvts (*this, nullptr, "Parameters", createParameterLayout())
{
    inputCircularBuffer.resize(8192, 0.0f);
    pitchHistory.resize(5, 0.0f);
}

AdaptiveAutoTuneAudioProcessor::~AdaptiveAutoTuneAudioProcessor()
{
}

const juce::String AdaptiveAutoTuneAudioProcessor::getName() const
{
    return "AdaptiveAutoTune";
}

bool AdaptiveAutoTuneAudioProcessor::acceptsMidi() const { return true; }
bool AdaptiveAutoTuneAudioProcessor::producesMidi() const { return false; }
bool AdaptiveAutoTuneAudioProcessor::isMidiEffect() const { return false; }
double AdaptiveAutoTuneAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int AdaptiveAutoTuneAudioProcessor::getNumPrograms() { return 1; }
int AdaptiveAutoTuneAudioProcessor::getCurrentProgram() { return 0; }
void AdaptiveAutoTuneAudioProcessor::setCurrentProgram (int index) {}
const juce::String AdaptiveAutoTuneAudioProcessor::getProgramName (int index) { return {}; }
void AdaptiveAutoTuneAudioProcessor::changeProgramName (int index, const juce::String& newName) {}

void AdaptiveAutoTuneAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    pitchShifters.resize (std::max (1, getTotalNumInputChannels()));
    for (auto& ps : pitchShifters)
    {
        ps.init (sampleRate);
        ps.clear();
    }
    std::fill (inputCircularBuffer.begin(), inputCircularBuffer.end(), 0.0f);
    std::fill (pitchHistory.begin(), pitchHistory.end(), 0.0f);
    circularBufferWritePos = 0;
    samplesSinceLastAnalysis = 0;
    smoothedCorrectionSemitones = 0.0f;
    lastTargetPitch = 220.0f;
    lastCandidateMidi = 0.0f;
    candidateConsecutiveFrames = 0;
    stableSnappedMidi = 0.0f;
    unvoicedConsecutiveFrames = 0;
}

void AdaptiveAutoTuneAudioProcessor::releaseResources()
{
}

bool AdaptiveAutoTuneAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void AdaptiveAutoTuneAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    int numSamples = buffer.getNumSamples();
    if (totalNumInputChannels == 0 || buffer.getNumChannels() == 0 || numSamples == 0)
        return;

    double sampleRate = getSampleRate();

    // Read parameter values from APVTS
    float amount = *apvts.getRawParameterValue ("amount");
    float speed = *apvts.getRawParameterValue ("speed");
    bool stateAware = *apvts.getRawParameterValue ("stateAware") > 0.5f;
    int scaleIndex = (int)(*apvts.getRawParameterValue ("scale"));
    int rootIndex = (int)(*apvts.getRawParameterValue ("root"));
    bool hardTune = *apvts.getRawParameterValue ("hardTune") > 0.5f;
    float confThresh = *apvts.getRawParameterValue ("confThresh");
    bool adaptiveRetune = *apvts.getRawParameterValue ("adaptiveRetune") > 0.5f;
    bool adaptiveWindow = *apvts.getRawParameterValue ("adaptiveWindow") > 0.5f;

    // We use Channel 0 for pitch tracking & framework analysis
    const float* inputChannelData = buffer.getReadPointer (0);

    // Feed circular buffer
    for (int i = 0; i < numSamples; ++i)
    {
        inputCircularBuffer[circularBufferWritePos] = inputChannelData[i];
        circularBufferWritePos = (circularBufferWritePos + 1) % (int)inputCircularBuffer.size();
    }

    samplesSinceLastAnalysis += numSamples;

    // Trigger analysis periodically
    if (samplesSinceLastAnalysis >= analysisHopSize)
    {
        samplesSinceLastAnalysis = 0;

        // Gather window size based on adaptive recommendation or default
        int curWindowSize = (stateAware && adaptiveWindow) ? activeWindowSize : 2048;
        
        // Extract linear frame from circular buffer ending at the current write position
        std::vector<float> analysisFrame (curWindowSize, 0.0f);
        for (int i = 0; i < curWindowSize; ++i)
        {
            int readPos = circularBufferWritePos - curWindowSize + i;
            while (readPos < 0) readPos += (int)inputCircularBuffer.size();
            analysisFrame[i] = inputCircularBuffer[readPos];
        }

        // Run C++ Engine analysis
        FrameworkState state = engine.analyze (analysisFrame.data(), curWindowSize, sampleRate);

        // Update atomics for UI thread
        z1.store (state.coordinate.first);
        z2.store (state.coordinate.second);
        safetySTFT.store (state.assumptions["stft"]);
        safetyACF.store (state.assumptions["acf"]);
        safetyCEP.store (state.assumptions["cepstrum"]);

        int rIdx = 0;
        if (state.region == "periodic_harmonic") rIdx = 1;
        else if (state.region == "noise_collapse") rIdx = 2;
        else if (state.region == "transient_overloaded") rIdx = 3;
        else if (state.region == "smooth_lowpass") rIdx = 4;
        regionIndex.store (rIdx);

        // Update adaptive parameters in the audio thread
        activeWindowSize = state.recommended_window;
        activeTroughThreshold = state.yin_trough;
        activeVoicingGate = state.voicing_gate;
        activeHoldPitch = state.hold_pitch;
    }

    // Determine current pitch and target correction
    float detectedF0 = 0.0f;
    int curWindowSize = (stateAware && adaptiveWindow) ? activeWindowSize : 2048;
    
    // Copy the latest analysis frame for the pitch tracker
    std::vector<float> pitchFrame (curWindowSize, 0.0f);
    float frameRms = 0.0f;
    for (int i = 0; i < curWindowSize; ++i)
    {
        int readPos = circularBufferWritePos - curWindowSize + i;
        while (readPos < 0) readPos += (int)inputCircularBuffer.size();
        pitchFrame[i] = inputCircularBuffer[readPos];
        frameRms += pitchFrame[i] * pitchFrame[i];
    }
    frameRms = std::sqrt (frameRms / curWindowSize);
    bool isSilent = (frameRms < 0.002f); // noise gate at -54dB for higher sensitivity

    if (stateAware && activeHoldPitch)
    {
        // Keep last valid pitch, bypass tracker
        detectedF0 = lastTargetPitch;
    }
    else if (isSilent)
    {
        detectedF0 = 0.0f;
    }
    else
    {
        float currentTrough = stateAware ? activeTroughThreshold : 0.15f;
        detectedF0 = pitchTracker.detectPitch (pitchFrame.data(), curWindowSize, sampleRate, currentTrough);
    }

    // Octave jump detection and correction relative to the last valid target pitch
    if (lastTargetPitch >= 80.0f && lastTargetPitch <= 1000.0f && detectedF0 >= 80.0f && detectedF0 <= 1000.0f)
    {
        float ratio = detectedF0 / lastTargetPitch;
        if (std::abs (ratio - 0.5f) < 0.08f)
        {
            detectedF0 *= 2.0f;
        }
        else if (std::abs (ratio - 2.0f) < 0.30f)
        {
            detectedF0 *= 0.5f;
        }
        else if (std::abs (ratio - 0.25f) < 0.04f)
        {
            detectedF0 *= 4.0f;
        }
        else if (std::abs (ratio - 4.0f) < 0.60f)
        {
            detectedF0 *= 0.25f;
        }
    }

    // 1. Determine raw voicing state
    bool rawIsVoiced = (detectedF0 >= 80.0f && detectedF0 <= 1000.0f && !isSilent);
    float currentConf = safetyACF.load();
    if (stateAware && currentConf < confThresh)
    {
        rawIsVoiced = false;
    }

    if (rawIsVoiced)
    {
        // Check if history is currently empty (all zeros)
        bool isHistoryEmpty = true;
        for (float val : pitchHistory)
        {
            if (val > 0.0f)
            {
                isHistoryEmpty = false;
                break;
            }
        }

        if (isHistoryEmpty)
        {
            std::fill (pitchHistory.begin(), pitchHistory.end(), detectedF0);
        }
        else
        {
            // Shift and append new voiced pitch
            for (size_t i = 0; i < pitchHistory.size() - 1; ++i)
            {
                pitchHistory[i] = pitchHistory[i + 1];
            }
            pitchHistory[pitchHistory.size() - 1] = detectedF0;
        }

        // Apply 5-point median filter to clean voiced estimates
        std::vector<float> sortedHistory = pitchHistory;
        std::sort (sortedHistory.begin(), sortedHistory.end());
        detectedF0 = sortedHistory[2];
    }

    // 2. Apply release gate hysteresis
    bool isVoiced = rawIsVoiced;
    if (rawIsVoiced)
    {
        unvoicedConsecutiveFrames = 0;
    }
    else
    {
        unvoicedConsecutiveFrames++;
        if (unvoicedConsecutiveFrames < 6 && lastTargetPitch >= 80.0f && lastTargetPitch <= 1000.0f)
        {
            isVoiced = true;
            detectedF0 = lastTargetPitch;
        }
        else
        {
            // Fully unvoiced/silent after hysteresis release: clear history
            std::fill (pitchHistory.begin(), pitchHistory.end(), 0.0f);
        }
    }

    float targetCorrection = 0.0f;

    if (isVoiced)
    {
        // Find nearest MIDI note in selected scale
        float rawMidi = 12.0f * std::log2 (detectedF0 / 440.0f) + 69.0f;
        float snappedMidi = snapToScale (rawMidi, scaleIndex, rootIndex);

        // Stabilize snapped MIDI note (debounce)
        if (snappedMidi == lastCandidateMidi)
        {
            candidateConsecutiveFrames++;
            if (candidateConsecutiveFrames >= 3)
            {
                stableSnappedMidi = snappedMidi;
            }
        }
        else
        {
            lastCandidateMidi = snappedMidi;
            candidateConsecutiveFrames = 1;
        }

        // Committed stable note, fallback to snappedMidi during initial latching
        float finalSnappedMidi = (stableSnappedMidi > 0.0f) ? stableSnappedMidi : snappedMidi;

        float targetF0 = 440.0f * std::pow (2.0f, (finalSnappedMidi - 69.0f) / 12.0f);
        
        targetCorrection = finalSnappedMidi - rawMidi;
        lastTargetPitch = targetF0;
    }
    else
    {
        // Reset debouncer and bypass/hold correction
        targetCorrection = 0.0f;
        lastCandidateMidi = 0.0f;
        candidateConsecutiveFrames = 0;
        stableSnappedMidi = 0.0f;
    }

    // Retune speed smoothing constant calculation (sample-level smoothing)
    float activeSpeed = speed;
    if (hardTune)
    {
        activeSpeed = 4.0f; // very fast snap but smoothed to prevent clicks
    }
    else if (stateAware && adaptiveRetune)
    {
        // Slow down retune speed in low-confidence regions (scale retune time constant up to 4x)
        activeSpeed = speed * (1.0f + 3.0f * (1.0f - currentConf));
    }
    float alphaSample = 1.0f - std::exp (-1.0f / (activeSpeed * sampleRate / 1000.0f));

    // Get write pointers for all channels to avoid calling getWritePointer inside sample loop
    std::vector<float*> channelPointers (totalNumInputChannels);
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
        channelPointers[channel] = buffer.getWritePointer (channel);
    }

    // Process audio buffer sample-by-sample, smoothing pitch shifts continuously
    for (int i = 0; i < numSamples; ++i)
    {
        smoothedCorrectionSemitones += alphaSample * (targetCorrection - smoothedCorrectionSemitones);
        float finalShift = smoothedCorrectionSemitones * (hardTune ? 1.0f : amount);

        for (int channel = 0; channel < totalNumInputChannels; ++channel)
        {
            if (channelPointers[channel] != nullptr && channel < (int)pitchShifters.size())
            {
                channelPointers[channel][i] = pitchShifters[channel].processSample (channelPointers[channel][i], finalShift, detectedF0);
            }
        }
    }

    static int debugCounter = 0;
    if (++debugCounter >= 100)
    {
        debugCounter = 0;
        std::ofstream logFile ("/Users/user/Desktop/representation-fragility-lab/scratch/debug_vst.txt", std::ios::app);
        if (logFile.is_open())
        {
            float finalShiftLog = smoothedCorrectionSemitones * (hardTune ? 1.0f : amount);
            logFile << "F0: " << detectedF0 
                    << " | Conf: " << currentConf 
                    << " | Shift: " << finalShiftLog 
                    << " | Voiced: " << (isVoiced ? "yes" : "no")
                    << " | Hard: " << (hardTune ? "yes" : "no")
                    << " | Scale: " << scaleIndex
                    << " | Root: " << rootIndex
                    << "\n";
        }
    }
}

juce::AudioProcessorEditor* AdaptiveAutoTuneAudioProcessor::createEditor()
{
    return new AdaptiveAutoTuneAudioProcessorEditor (*this);
}

bool AdaptiveAutoTuneAudioProcessor::hasEditor() const { return true; }

void AdaptiveAutoTuneAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AdaptiveAutoTuneAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (apvts.state.getType()))
            apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}

juce::AudioProcessorValueTreeState::ParameterLayout AdaptiveAutoTuneAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Basic controls
    params.push_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID ("amount", 1), "Correction Amount", 0.0f, 1.0f, 1.0f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID ("speed", 1), "Retune Speed (ms)", 1.0f, 500.0f, 30.0f));
    params.push_back (std::make_unique<juce::AudioParameterBool> (juce::ParameterID ("stateAware", 1), "State-Aware Mode", true));

    // Scale selector choice parameter
    juce::StringArray scaleChoices = {
        "Chromatic",
        "Major",
        "Natural Minor",
        "Pentatonic Major",
        "Pentatonic Minor"
    };
    params.push_back (std::make_unique<juce::AudioParameterChoice> (juce::ParameterID ("scale", 1), "Scale Key/Type", scaleChoices, 0));

    // Root Note choice parameter
    juce::StringArray rootChoices = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    params.push_back (std::make_unique<juce::AudioParameterChoice> (juce::ParameterID ("root", 1), "Root Note", rootChoices, 0));
    params.push_back (std::make_unique<juce::AudioParameterBool> (juce::ParameterID ("hardTune", 1), "Hard Autotune", false));

    // Advanced controls
    params.push_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID ("confThresh", 1), "Confidence Threshold", 0.0f, 1.0f, 0.40f));
    params.push_back (std::make_unique<juce::AudioParameterBool> (juce::ParameterID ("adaptiveRetune", 1), "Adaptive Retune Speed", true));
    params.push_back (std::make_unique<juce::AudioParameterBool> (juce::ParameterID ("adaptiveWindow", 1), "Adaptive Window Size", true));

    return { params.begin(), params.end() };
}

// This creates the processor
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdaptiveAutoTuneAudioProcessor();
}
