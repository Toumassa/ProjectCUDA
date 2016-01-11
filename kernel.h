#ifndef DEF_KERNEL
#define DEF_KERNEL


#include "RandomForest.h"
#include "GPUAdapter.h"
/* Cuda stuff */


/*
float gpuGetValue (float *gpuFeatures, uint8_t channel, 
    int16_t x, int16_t y, int16_t w, int16_t h);

__device__
float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
    int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h);


__device__
SplitResult split(SplitData<float> splitData, Sample<float> &sample, int16_t w, int16_t h, int16_t w_i, int16_t h_i, float *gpuFeatures, float *gpuFeaturesIntegral);

__global__
void predict(int *returnStartHistTab, int *returnCountHistTab, ANode* tree, int16_t w, int16_t h, int16_t w_i, int16_t h_i, float* features, float* features_integral, Sample<float> &sample);


void copyTreeToGPU(ANode *, ANode**, int);
void copyFeaturesToGPU(float *features, int fsize, 
						float *integral_features, int fIntegralSize,
						float **_features,
						float **_integral_features
						);
void copyCommonHistTabToGPU(uint32_t*, uint32_t**, int);*/
#endif
