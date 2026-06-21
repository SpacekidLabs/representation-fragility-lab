#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <vector>
#include <algorithm>

static float snapToScale (float rawMidiNote, int scaleIndex)
{
    if (scaleIndex == 0) // Chromatic
        return std::round (rawMidiNote);

    // Get valid pitch classes for the selected scale
    std::vector<int> validClasses;
    switch (scaleIndex)
    {
        case 1: validClasses = { 0, 2, 4, 5, 7, 9, 11 }; break; // C Major
        case 2: validClasses = { 0, 2, 3, 5, 7, 8, 10 }; break; // C Minor
        case 3: validClasses = { 0, 2, 4, 7, 9 };         break; // C Pentatonic Major
        case 4: validClasses = { 0, 3, 5, 7, 10 };        break; // C Pentatonic Minor
        case 5: validClasses = { 0, 2, 4, 6, 7, 9, 11 }; break; // G Major (contains F# = 6)
        case 6: validClasses = { 0, 2, 4, 5, 7, 9, 11 }; break; // A Minor (same as C Major)
        default: return std::round (rawMidiNote);
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
    circularBufferWritePos = 0;
    samplesSinceLastAnalysis = 0;
    smoothedCorrectionSemitones = 0.0f;
    lastTargetPitch = 220.0f;
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
    for (int i = 0; i < curWindowSize; ++i)
    {
        int readPos = circularBufferWritePos - curWindowSize + i;
        while (readPos < 0) readPos += (int)inputCircularBuffer.size();
        pitchFrame[i] = inputCircularBuffer[readPos];
    }

    if (stateAware && activeHoldPitch)
    {
        // Keep last valid pitch, bypass tracker
        detectedF0 = lastTargetPitch;
    }
    else
    {
        float currentTrough = stateAware ? activeTroughThreshold : 0.15f;
        detectedF0 = pitchTracker.detectPitch (pitchFrame.data(), curWindowSize, sampleRate, currentTrough);
    }

    float targetCorrection = 0.0f;

    // Apply confidence gate based on ACF safety score
    float currentConf = safetyACF.load();
    bool isVoiced = (detectedF0 >= 80.0f && detectedF0 <= 1000.0f);
    if (stateAware && currentConf < confThresh)
    {
        isVoiced = false;
    }

    if (isVoiced)
    {
        // Find nearest MIDI note in selected scale
        float rawMidi = 12.0f * std::log2 (detectedF0 / 440.0f) + 69.0f;
        float snappedMidi = snapToScale (rawMidi, scaleIndex);
        float targetF0 = 440.0f * std::pow (2.0f, (snappedMidi - 69.0f) / 12.0f);
        
        targetCorrection = snappedMidi - rawMidi;
        lastTargetPitch = targetF0;
    }
    else
    {
        // Hold pitch or bypass correction
        targetCorrection = 0.0f;
    }

    // Apply voicing gate to mute correction in noise/silence
    if (stateAware && activeVoicingGate && !isVoiced)
    {
        targetCorrection = 0.0f;
    }

    // Retune speed smoothing constant calculation
    float activeSpeed = speed;
    if (stateAware && adaptiveRetune)
    {
        // Slow down retune speed in low-confidence regions (scale retune time constant up to 4x)
        activeSpeed = speed * (1.0f + 3.0f * (1.0f - currentConf));
    }

    float alpha = 1.0f - std::exp (-numSamples / (activeSpeed * sampleRate / 1000.0f));
    smoothedCorrectionSemitones += alpha * (targetCorrection - smoothedCorrectionSemitones);

    // Apply Correction Amount scalar
    float finalShift = smoothedCorrectionSemitones * amount;

    // Process audio buffer sample-by-sample (multi-channel independent pitch shifters)
    for (int channel = 0; channel < totalNumInputChannels; ++channel)
    {
        float* channelData = buffer.getWritePointer (channel);
        
        if (channel < (int)pitchShifters.size())
        {
            if (std::abs(finalShift) < 0.01f)
            {
                pitchShifters[channel].clear();
            }

            for (int i = 0; i < numSamples; ++i)
            {
                channelData[i] = pitchShifters[channel].processSample (channelData[i], finalShift);
            }
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
        "C Major",
        "C Natural Minor",
        "C Major Pentatonic",
        "C Minor Pentatonic",
        "G Major",
        "A Natural Minor"
    };
    params.push_back (std::make_unique<juce::AudioParameterChoice> (juce::ParameterID ("scale", 1), "Scale", scaleChoices, 0));

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
