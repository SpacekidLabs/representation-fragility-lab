#include "PluginProcessor.h"
#include "PluginEditor.h"

// -----------------------------------------------------------------------------
// State Space Visualizer Implementation
// -----------------------------------------------------------------------------
StateSpaceVisualizer::StateSpaceVisualizer(AdaptiveAutoTuneAudioProcessor& p)
    : processor (p)
{
    startTimerHz(30); // 30 FPS polling
}

StateSpaceVisualizer::~StateSpaceVisualizer()
{
    stopTimer();
}

void StateSpaceVisualizer::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();
    
    // Background fill
    g.setColour(juce::Colour::fromString("#0D1117"));
    g.fillRoundedRectangle(bounds, 8.0f);
    
    // Draw boundary border
    g.setColour(juce::Colour::fromString("#30363D"));
    g.drawRoundedRectangle(bounds, 8.0f, 1.5f);

    float w = bounds.getWidth();
    float h = bounds.getHeight();
    float margin = 10.0f;
    
    // Coordinate mapping helper lambda
    // Mapping: z1 in [-3.0, 6.0], z2 in [-3.0, 3.0]
    auto getPixelCoords = [=](float z1Val, float z2Val) -> juce::Point<float> {
        float px = margin + ((z1Val - (-3.0f)) / 9.0f) * (w - 2.0f * margin);
        float py = h - margin - ((z2Val - (-3.0f)) / 6.0f) * (h - 2.0f * margin);
        return { px, py };
    };

    // 1. Draw region background shadings
    // Periodic Harmonic Region: z1 < -0.5, z2 < -0.2
    auto ptTopLeftPH = getPixelCoords(-3.0f, 3.0f);
    auto ptBottomRightPH = getPixelCoords(-0.5f, -0.2f);
    auto ptBottomEdgePH = getPixelCoords(-0.5f, -3.0f);
    g.setColour(juce::Colours::green.withAlpha(0.06f));
    g.fillRect(ptTopLeftPH.x, ptTopLeftPH.y, ptBottomRightPH.x - ptTopLeftPH.x, ptBottomEdgePH.y - ptTopLeftPH.y);

    // Noise Collapse Region: z1 > 1.5
    auto ptTopLeftNC = getPixelCoords(1.5f, 3.0f);
    auto ptBottomRightNC = getPixelCoords(6.0f, -3.0f);
    g.setColour(juce::Colours::red.withAlpha(0.06f));
    g.fillRect(ptTopLeftNC.x, ptTopLeftNC.y, ptBottomRightNC.x - ptTopLeftNC.x, ptBottomRightNC.y - ptTopLeftNC.y);

    // Transient Overloaded Region: z2 > 1.5, z1 <= 1.5
    auto ptTopLeftTO = getPixelCoords(-3.0f, 3.0f);
    auto ptBottomRightTO = getPixelCoords(1.5f, 1.5f);
    g.setColour(juce::Colours::cyan.withAlpha(0.06f));
    g.fillRect(ptTopLeftTO.x, ptTopLeftTO.y, ptBottomRightTO.x - ptTopLeftTO.x, ptBottomRightTO.y - ptTopLeftTO.y);

    // Smooth Lowpass Region: -0.5 <= z1 <= 1.5, z2 < -0.2
    auto ptTopLeftSLP = getPixelCoords(-0.5f, -0.2f);
    auto ptBottomRightSLP = getPixelCoords(1.5f, -3.0f);
    g.setColour(juce::Colours::blueviolet.withAlpha(0.06f));
    g.fillRect(ptTopLeftSLP.x, ptTopLeftSLP.y, ptBottomRightSLP.x - ptTopLeftSLP.x, ptBottomRightSLP.y - ptTopLeftSLP.y);

    // 2. Draw axis lines
    auto origin = getPixelCoords(0.0f, 0.0f);
    g.setColour(juce::Colour::fromString("#21262D").withAlpha(0.5f));
    g.drawHorizontalLine((int)origin.y, margin, w - margin);
    g.drawVerticalLine((int)origin.x, margin, h - margin);

    // 3. Draw region text labels
    g.setFont(9.0f);
    g.setColour(juce::Colours::white.withAlpha(0.25f));
    
    auto drawLabel = [&](const juce::String& text, float z1Pos, float z2Pos) {
        auto pt = getPixelCoords(z1Pos, z2Pos);
        g.drawText(text, (int)pt.x - 50, (int)pt.y - 6, 100, 12, juce::Justification::centred, false);
    };

    drawLabel("Harmonic", -1.8f, 1.8f);
    drawLabel("Noise", 3.8f, 1.8f);
    drawLabel("Transient", -0.8f, 2.2f);
    drawLabel("Lowpass", 0.5f, -1.8f);
    drawLabel("Transition", 0.5f, 0.5f);

    // 4. Draw smooth real-time tracking dot
    auto dotPt = getPixelCoords(smoothX, smoothY);
    
    // Outer glow
    juce::Graphics::ScopedSaveState state(g);
    g.setColour(juce::Colours::orange.withAlpha(0.20f));
    g.fillEllipse(dotPt.x - 12.0f, dotPt.y - 12.0f, 24.0f, 24.0f);
    
    // Core dot
    g.setColour(juce::Colours::orange);
    g.fillEllipse(dotPt.x - 4.5f, dotPt.y - 4.5f, 9.0f, 9.0f);
}

void StateSpaceVisualizer::timerCallback()
{
    // Retrieve coordinates from audio thread
    float targetX = processor.z1.load();
    float targetY = processor.z2.load();
    
    // Glide smoothing
    smoothX += 0.22f * (targetX - smoothX);
    smoothY += 0.22f * (targetY - smoothY);

    // Clip display boundary limits
    smoothX = std::clamp(smoothX, -3.0f, 6.0f);
    smoothY = std::clamp(smoothY, -3.0f, 3.0f);

    repaint();
}

// -----------------------------------------------------------------------------
// Main Plugin Editor Implementation
// -----------------------------------------------------------------------------
AdaptiveAutoTuneAudioProcessorEditor::AdaptiveAutoTuneAudioProcessorEditor(AdaptiveAutoTuneAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p), visualizer (p)
{
    // Sliders
    amountSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    amountSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 50, 16);
    addAndMakeVisible (amountSlider);

    speedSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    speedSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 50, 16);
    addAndMakeVisible (speedSlider);

    confThreshSlider.setSliderStyle (juce::Slider::LinearBar);
    confThreshSlider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    addChildComponent (confThreshSlider); // hidden inside Advanced by default

    // Toggles
    stateAwareToggle.setButtonText ("State-Aware Tracking");
    addAndMakeVisible (stateAwareToggle);

    adaptiveRetuneToggle.setButtonText ("Adaptive Retune Speed");
    addChildComponent (adaptiveRetuneToggle);

    adaptiveWindowToggle.setButtonText ("Adaptive Window Scaling");
    addChildComponent (adaptiveWindowToggle);

    hardTuneToggle.setButtonText ("Hard Autotune (T-Pain)");
    addAndMakeVisible (hardTuneToggle);

    // Advanced Trigger Button
    advancedButton.setButtonText ("Advanced Tools ▼");
    advancedButton.addListener (this);
    addAndMakeVisible (advancedButton);

    // Scale Selector ComboBox
    scaleSelector.addItem ("Chromatic", 1);
    scaleSelector.addItem ("C Major", 2);
    scaleSelector.addItem ("C Natural Minor", 3);
    scaleSelector.addItem ("C Major Pentatonic", 4);
    scaleSelector.addItem ("C Minor Pentatonic", 5);
    scaleSelector.addItem ("G Major", 6);
    scaleSelector.addItem ("A Natural Minor", 7);
    addAndMakeVisible (scaleSelector);

    // Attachments
    amountAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, "amount", amountSlider);
    speedAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, "speed", speedSlider);
    confThreshAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (audioProcessor.apvts, "confThresh", confThreshSlider);

    stateAwareAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "stateAware", stateAwareToggle);
    adaptiveRetuneAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "adaptiveRetune", adaptiveRetuneToggle);
    adaptiveWindowAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "adaptiveWindow", adaptiveWindowToggle);

    scaleAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (audioProcessor.apvts, "scale", scaleSelector);
    hardTuneAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (audioProcessor.apvts, "hardTune", hardTuneToggle);

    // Add state space visualizer
    addAndMakeVisible (visualizer);

    // Size configuration (standalone app default)
    setSize (640, 360);
}

AdaptiveAutoTuneAudioProcessorEditor::~AdaptiveAutoTuneAudioProcessorEditor()
{
    advancedButton.removeListener (this);
}

void AdaptiveAutoTuneAudioProcessorEditor::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    
    // Background Dark Gradient
    juce::ColourGradient gradient(juce::Colour::fromString("#161B22"), 0.0f, 0.0f,
                                  juce::Colour::fromString("#0D1117"), 0.0f, (float)bounds.getHeight(), false);
    g.setGradientFill (gradient);
    g.fillAll();

    // 1. Title bar
    g.setColour (juce::Colours::white);
    g.setFont (juce::Font("sans-serif", 16.0f, juce::Font::bold));
    g.drawText ("ADAPTIVE AUTO-TUNE", 20, 18, 300, 24, juce::Justification::left, true);

    g.setColour (juce::Colours::white.withAlpha(0.3f));
    g.setFont (juce::Font("sans-serif", 10.0f, juce::Font::plain));
    g.drawText ("Representation Intelligence V1", 20, 38, 250, 12, juce::Justification::left, true);

    // 2. Indicators & Region Details (Middle Right)
    int startY = 75;
    int colX = 330;

    g.setColour (juce::Colours::white);
    g.setFont (juce::Font("sans-serif", 12.0f, juce::Font::bold));
    g.drawText ("Signal Region State:", colX, startY, 200, 16, juce::Justification::left, true);

    // Read atomic values
    int rIdx = audioProcessor.regionIndex.load();
    juce::String regionStr = "Transition Zone";
    juce::Colour regionColor = juce::Colour::fromString("#8B949E"); // Grey
    
    if (rIdx == 1) { regionStr = "Periodic Harmonic"; regionColor = juce::Colour::fromString("#2EA44F"); }
    else if (rIdx == 2) { regionStr = "Noise Collapse"; regionColor = juce::Colour::fromString("#F85149"); }
    else if (rIdx == 3) { regionStr = "Transient Burst"; regionColor = juce::Colour::fromString("#58A6FF"); }
    else if (rIdx == 4) { regionStr = "Smooth Lowpass"; regionColor = juce::Colour::fromString("#BC8CFF"); }

    g.setColour (regionColor);
    g.setFont (juce::Font("sans-serif", 14.0f, juce::Font::bold));
    g.drawText (regionStr, colX, startY + 20, 260, 20, juce::Justification::left, true);

    // Representation Safety Checkboxes
    int checkY = startY + 55;
    g.setColour (juce::Colours::white.withAlpha(0.6f));
    g.setFont (10.0f);
    g.drawText ("Representation Safety Bounds:", colX, checkY, 200, 12, juce::Justification::left, true);

    auto drawIndicator = [&](const juce::String& name, float score, int yPos) {
        g.setColour (juce::Colours::white);
        g.setFont (10.5f);
        g.drawText (name, colX + 16, yPos, 80, 14, juce::Justification::left, true);

        // Green light for safe (>0.5), red for collapsed (<0.5)
        bool isSafe = (score >= 0.50f);
        g.setColour (isSafe ? juce::Colour::fromString("#2EA44F") : juce::Colour::fromString("#F85149"));
        g.fillEllipse ((float)colX + 2.0f, (float)yPos + 3.0f, 7.0f, 7.0f);
    };

    drawIndicator ("ACF (Periodic)", audioProcessor.safetyACF.load(), checkY + 18);
    drawIndicator ("STFT (Stationary)", audioProcessor.safetySTFT.load(), checkY + 36);
    drawIndicator ("CEP (Harmonics)", audioProcessor.safetyCEP.load(), checkY + 54);

    // Scale Selector Label
    g.setColour (juce::Colours::white.withAlpha(0.7f));
    g.setFont (juce::Font("sans-serif", 10.5f, juce::Font::bold));
    g.drawText ("Scale Key/Type:", colX, 263, 140, 14, juce::Justification::left, true);

    // Advanced Section Label
    if (isAdvancedOpen)
    {
        g.setColour (juce::Colours::white.withAlpha(0.4f));
        g.setFont (11.0f);
        g.drawText ("Advanced Coordination Parameters:", 20, 362, 300, 14, juce::Justification::left, true);
        
        g.setColour (juce::Colours::white);
        g.setFont (9.5f);
        g.drawText ("Gating Conf Threshold:", 420, 375, 120, 12, juce::Justification::left, true);
        
        float thresh = audioProcessor.apvts.getRawParameterValue("confThresh")->load();
        g.drawText (juce::String(thresh, 2), 540, 375, 40, 12, juce::Justification::left, true);
    }
}

void AdaptiveAutoTuneAudioProcessorEditor::resized()
{
    // Layout boundaries
    visualizer.setBounds (20, 70, 270, 270);

    // Position dials on the far right
    int dialX = 490;
    amountSlider.setBounds (dialX, 75, 110, 110);
    speedSlider.setBounds (dialX, 205, 110, 110);

    // Position main toggles
    hardTuneToggle.setBounds (330, 205, 150, 24);
    stateAwareToggle.setBounds (330, 235, 150, 24);
    scaleSelector.setBounds (330, 280, 140, 24);
    advancedButton.setBounds (330, 315, 140, 26);

    // Display Advanced options dynamically
    if (isAdvancedOpen)
    {
        confThreshSlider.setVisible (true);
        adaptiveRetuneToggle.setVisible (true);
        adaptiveWindowToggle.setVisible (true);

        confThreshSlider.setBounds (420, 395, 180, 20);
        adaptiveRetuneToggle.setBounds (20, 390, 180, 24);
        adaptiveWindowToggle.setBounds (20, 420, 180, 24);
    }
    else
    {
        confThreshSlider.setVisible (false);
        adaptiveRetuneToggle.setVisible (false);
        adaptiveWindowToggle.setVisible (false);
    }
}

void AdaptiveAutoTuneAudioProcessorEditor::buttonClicked(juce::Button* button)
{
    if (button == &advancedButton)
    {
        isAdvancedOpen = !isAdvancedOpen;
        advancedButton.setButtonText (isAdvancedOpen ? "Advanced Tools ▲" : "Advanced Tools ▼");
        
        // Resize editor window height
        setSize (640, isAdvancedOpen ? 465 : 360);
        resized();
        repaint();
    }
}
