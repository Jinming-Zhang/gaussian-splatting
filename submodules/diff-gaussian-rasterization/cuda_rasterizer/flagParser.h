#ifndef CUDA_FLAGPARSER_H_INCLUDED
#define CUDA_FLAGPARSER_H_INCLUDED
__global__ void parseRenderMode(int flag, int *out);
__global__ void parseLearningMode(int flag, int *out);

#endif
