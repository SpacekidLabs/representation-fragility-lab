#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

// -----------------------------------------------------------------------------
// State Space Visualizer Component
// -----------------------------------------------------------------------------
class StateSpaceVisualizer : public juce::Component, public juce::Timer
{
public:
    StateSpaceVisualizer (AdaptiveAutoTuneAudioProcessor& p);
    ~StateSpaceVisualizer() override;

    void paint (juce::Graphics& g) override;
    void timerCallback() override;

private:
    AdaptiveAutoTuneAudioProcessor& processor;
    
    // Smooth visual dot interpolation
    float smoothX { 0.0f };
    float smoothY { 0.0f };
};

// -----------------------------------------------------------------------------
// Main Plugin Editor Component
// -----------------------------------------------------------------------------
class AdaptiveAutoTuneAudioProcessorEditor  : public juce::AudioProcessorEditor,
                                              public juce::Button::Listener
{
public:
    AdaptiveAutoTuneAudioProcessorEditor (AdaptiveAutoTuneAudioProcessor&);
    ~AdaptiveAutoTuneAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;
    void buttonClicked (juce::Button* button) override;

private:
    AdaptiveAutoTuneAudioProcessor& audioProcessor;

    // Controls
    juce::Slider amountSlider;
    juce::Slider speedSlider;
    juce::Slider confThreshSlider;

    juce::ToggleButton stateAwareToggle;
    juce::ToggleButton adaptiveRetuneToggle;
    juce::ToggleButton adaptiveWindowToggle;

    juce::TextButton advancedButton;
    bool isAdvancedOpen { false };

    // Attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> amountAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> speedAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> confThreshAttachment;

    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> stateAwareAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> adaptiveRetuneAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> adaptiveWindowAttachment;

    // State space plot component
    StateSpaceVisualizer visualizer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveAutoTuneAudioProcessorEditor)
};
