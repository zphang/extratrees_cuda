#ifndef MNISTPREPROCESS_H
#define MNISTPREPROCESS_H
#include <stdio.h>

static inline void revertInt(int*x)
{
	*x=((*x&0x000000ff)<<24)|((*x&0x0000ff00)<<8)|((*x&0x00ff0000)>>8)|((*x&0xff000000)>>24);
};
void readData(float* dataset,float *labels,const char* dataPath,const char*labelPath);
#endif//MNISTPREPROCESS_H