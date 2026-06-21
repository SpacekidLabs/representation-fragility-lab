#pragma once
#include <vector>
#include <cmath>

namespace juce {
namespace dsp {
class FFT {
public:
    FFT(int order) : size(1 << order) {}
    ~FFT() = default;
    
    void performRealOnlyForwardTransform(float* buffer) {
        // buffer has size 2 * size
        std::vector<float> in(size);
        for (int i = 0; i < size; ++i) in[i] = buffer[i];
        
        // Compute real-only DFT:
        float dc = 0.0f;
        for (int n = 0; n < size; ++n) dc += in[n]; // DC
        
        float nyquist = 0.0f;
        for (int n = 0; n < size; ++n) {
            nyquist += in[n] * ((n % 2 == 0) ? 1.0f : -1.0f); // Nyquist
        }

        buffer[0] = dc;
        buffer[1] = nyquist;
        
        for (int k = 1; k < size / 2; ++k) {
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < size; ++n) {
                float angle = 2.0f * M_PI * k * n / size;
                re += in[n] * std::cos(angle);
                im -= in[n] * std::sin(angle);
            }
            buffer[2 * k] = re;
            buffer[2 * k + 1] = im;
        }
    }
private:
    int size;
};
}
}
