#include "kernel.h"



void copyTreeToGPU(ANode *cpuTree, ANode**gpuTree, int treeSize)
{
	cudaError_t ok;
	size_t size;
	
	// Allocate GPU memory for the features and transfer
	// them from host memory to GPU memory
	size=treeSize*sizeof(ANode);
	ok=cudaMalloc ((void**) gpuTree, size);
	
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu allocation for tree:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	//CHECK_CUDA_MALLOC;
	ok=cudaMemcpy (*gpuTree, cpuTree, size, cudaMemcpyHostToDevice);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error memcpy RAM to GPU for tree storage\n";
		exit(1);
	}
	
	
}


void copyFeaturesToGPU(float *features, int fsize, float *integral_features, int fintegral_size, float **_features, float **_integral_features)
{
	cudaError_t ok;
	size_t size;
	
	size = fsize*sizeof(float);
	ok=cudaMalloc ((void**) _features, size);
	
	
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu allocation for features:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	ok=cudaMemcpy (*_features, features, size, cudaMemcpyHostToDevice);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu memcpy for features:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	
	size = fintegral_size*sizeof(float);
	ok=cudaMalloc ((void**) _integral_features, size);
	
	
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu allocation for features_integral:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	ok=cudaMemcpy (*_integral_features, integral_features, size, cudaMemcpyHostToDevice);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu memcpy for features_integral:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
}


void copyCommonHistTabToGPU(uint32_t*hist, uint32_t**_hist, int hsize)
{
	cudaError_t ok;
	size_t size;
	
	size = hsize*sizeof(uint32_t);
	ok=cudaMalloc ((void**) _hist, size);
	
	
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu allocation for features:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	ok=cudaMemcpy (*_hist, hist, size, cudaMemcpyHostToDevice);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu memcpy for features:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	
	ok=cudaMemcpy (hist,*_hist, size, cudaMemcpyDeviceToHost);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu memcpy for features:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	/*for(int i = 0; i < hsize; i++)
	cout << "*h=" << hist[i] <<endl;*/
}

void GPUAdapter::PushTreeToGPU(int n)
{
	if(n < 0 || n > this->treesAsVector.size())
	{
		cerr << "PushTreeToGPU: incorrect tree index" << endl;
		exit(1);
	}

	//change with malloc GPU
	ANode *treeToGPU;
	
	/*for(int i = 0; i < this->treesAsVector[n]->size();i++)
		tree[i]=(*this->treesAsVector[n])[i];*/
	
	copyTreeToGPU(this->treesAsVector[n]->data(), &treeToGPU, this->treesAsVector[n]->size());
	
	_treeAsTab.push_back(treeToGPU);
}

__device__
float gpuGetValue (float *gpuFeatures, uint8_t channel, 
    int16_t x, int16_t y, int16_t w, int16_t h)
{
  //cout << "before gpuGetValue\n";
    float res = gpuFeatures[y+x*h + channel*w*h];
    return res;
}

__device__
float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
    int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h)
{
    float res = (
            gpuFeaturesIntegral[y2 + x2*h + channel*w*h] -
            gpuFeaturesIntegral[y2 + x1*h + channel*w*h] -
            gpuFeaturesIntegral[y1 + x2*h + channel*w*h] +
            gpuFeaturesIntegral[y1 + x1*h + channel*w*h]);

    return res;
}


__device__
SplitResult split(SplitData<float> splitData, Sample<float> &sample, int16_t w, int16_t h, int16_t w_i, int16_t h_i, float *gpuFeatures, float *gpuFeaturesIntegral)
{
   sample.value = gpuGetValue(gpuFeatures, splitData.channel0, sample.x, sample.y, w, h);
	SplitResult centerResult = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;
    if (splitData.fType == 0) // single probe (center only)
    {
        return centerResult;
    }
    // for cases when we have non-centered probe types
    int pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y;

    pt1x = sample.x + splitData.dx1 - splitData.bw1;
    pt1y = sample.y + splitData.dy1 - splitData.bh1;

    pt2x = sample.x + splitData.dx1 + splitData.bw1 + 1; // remember -> integral images have size w+1 x h+1
    pt2y = sample.y + splitData.dy1 + splitData.bh1 + 1;

    if (pt1x < 0 || pt2x < 0 || pt1y < 0 || pt2y < 0 ||
        pt1x > w || pt2x > w || pt1y > h || pt2y > h) // due to size correction in getImgXXX we dont have to check \geq
    {
      return centerResult;
    }
    else
    {
      if (splitData.fType == 1) // single probe (center - offset)
      {
        int16_t norm1 = (pt2x - pt1x) * (pt2y - pt1y);
        sample.value -= gpuGetValueIntegral(gpuFeaturesIntegral, splitData.channel0, pt1x, pt1y, pt2x, pt2y, w_i, h_i) / norm1;
      }
      else                      // pixel pair probe test
      {
        pt3x = sample.x + splitData.dx2 - splitData.bw2;
        pt3y = sample.y + splitData.dy2 - splitData.bh2;

        pt4x = sample.x + splitData.dx2 + splitData.bw2 + 1;
        pt4y = sample.y + splitData.dy2 + splitData.bh2 + 1;


        if (pt3x < 0 || pt4x < 0 || pt3y < 0 || pt4y < 0 ||
            pt3x > w || pt4x > w || pt3y > h || pt4y > h)
        {
          return centerResult;
        }

        int16_t norm1 = (pt2x - pt1x) * (pt2y - pt1y);
        int16_t norm2 = (pt4x - pt3x) * (pt4y - pt3y);

        if (splitData.fType == 2)    // sum of pair probes
        {
          sample.value = gpuGetValueIntegral(gpuFeaturesIntegral, splitData.channel0, pt1x, pt1y, pt2x, pt2y, w_i, h_i) / norm1
                       + gpuGetValueIntegral(gpuFeaturesIntegral, splitData.channel1, pt3x, pt3y, pt4x, pt4y, w_i, h_i) / norm2;
        }
        else if (splitData.fType == 3)  // difference of pair probes
        {
          sample.value = gpuGetValueIntegral(gpuFeaturesIntegral, splitData.channel0, pt1x, pt1y, pt2x, pt2y, w_i, h_i) / norm1
                       - gpuGetValueIntegral(gpuFeaturesIntegral, splitData.channel1, pt3x, pt3y, pt4x, pt4y, w_i, h_i) / norm2;
        }

      }
    }
    SplitResult res = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;

    return res;
}
__device__
void predict(int *returnStartHistTab, ANode* tree, int16_t w, int16_t h, int16_t w_i, int16_t h_i, float* features, float* features_integral, Sample<float> sample)
{
  int curNode = 0; //initialising to Root
    SplitResult sr = SR_LEFT;
    while (tree[curNode].left != -1 && sr != SR_INVALID)
    {
    sr = split(tree[curNode].splitData, sample, w, h, w_i, h_i, features, features_integral);
   
    switch (sr)
      {
      case SR_LEFT:
        curNode = tree[curNode].left;
        break;
      case SR_RIGHT:
        curNode = tree[curNode].right;
        break;
      default:
        break;
      }
    }
    (*returnStartHistTab) = tree[curNode].common_hist_tab_offset;
}
/**
	Copies data which doesnt depend on images
*/
void GPUAdapter::init(StrucClassSSF<float> *forest, ConfigReader *cr)
{
	this->treeTabCount = cr->numTrees;
	for(size_t t = 0; t < this->treeTabCount; ++t)
    {
    	this->AddTree(&(forest[t]));
    }

	for(int i = 0; i < this->treeTabCount; i++)
    {
    	PushTreeToGPU(i);
    }
	
	copyCommonHistTabToGPU(common_hist_tab.data(), &_common_hist_tab, this->common_hist_tab.size());
	
	common_hist_tab.clear();
}
void GPUAdapter::preKernel(uint16_t imageId, ConfigReader *cr, TrainingSetSelection<float> *pTS)
{
	cudaError_t ok;
    this->ts = pTS;
    
    this->pImageData = this->ts->pImageData;
    this->nChannels = this->ts->getNChannels();
    this->iWidth = this->ts->getImgWidth(0);
    this->iHeight = this->ts->getImgHeight(0);
    this->numLabels = cr->numLabels;
    this->lPXOff = cr->labelPatchWidth / 2;
    this->lPYOff = cr->labelPatchHeight / 2;

	
  
    
    this->getFlattenedFeatures(imageId, &(this->features), &(this->nChannels));
    this->getFlattenedIntegralFeatures(imageId, &(this->features_integral), &(this->w_integral), &(this->h_integral));

    copyFeaturesToGPU(this->features, this->fSize, 
						this->features_integral, this->fIntegralSize, 
						&this->_features, &this->_features_integral);
	
	
	int size = this->iWidth*this->iHeight*this->numLabels*sizeof(int);
	this->result = (int*)malloc(size);
	
	/*for(int i =0; i < this->iWidth*this->iHeight*this->numLabels; i++)
	{
		this->result[i] = 0;
	}*/
    
	ok = cudaMalloc((void**) &this->resultGPU, size);if(ok != cudaSuccess)
	{
		std::cerr << "Error gpu allocation this->resultGPU:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	ok= cudaMemset  ( this->resultGPU,
                            0,
                           size    
                        )  ;
	if(ok != cudaSuccess)
	{
		std::cerr << "Error gpuresult initializing:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
	ok=cudaMemcpy (this->resultGPU, this->result, size, cudaMemcpyHostToDevice);
	if(ok != cudaSuccess)
	{
		std::cerr << "Error memcpy RAM to GPU for tree this->result\n";
		exit(1);
	}
	if(this->result == NULL)
	{
		std::cerr << " HOST memory allocation failed" << endl;
		exit(1);
	}
	delete [] this->features;
	delete [] this->features_integral;
}

__global__
void kernel(int *result, ANode* tree, int16_t w, int16_t h, int16_t w_i, int16_t h_i, float* features, float* features_integral, 
		Sample<float> sample, int lPXOff, int lPYOff, uint32_t *common_hist_tab, int numLabels)
{
	int sx = blockIdx.x*blockDim.x+threadIdx.x;
	int sy = blockIdx.y*blockDim.y+threadIdx.y;
	
	int p;
	sample.x = sx;
	sample.y = sy;
	predict(&p, tree, w, h, w_i, h_i, features, features_integral, sample);
	
	int ptx, pty;
	for (pty=(int)sy-lPYOff;pty<=(int)sy+(int)lPYOff;++pty)
	for (ptx=(int)sx-(int)lPXOff;ptx<=(int)sx+(int)lPXOff;++ptx,++p)
	{
		if (common_hist_tab[p]< 0 || common_hist_tab[p] >= numLabels)
		{
			/*cout << "x:" << sx << " y:"<<sy << " tree:"<< t << endl;
			cout << "pt.x:" << pt.x << " pt.y:"<<pt.y << ":"<< p << endl;
			cout << "*p : " << common_hist_tab[p] << endl;
			//std::cerr << "Invalid label in prediction: " << (int) common_hist_tab[p] << "\n";
			*///exit(1);
		}         
		else if (ptx >=0 && ptx<w && pty >= 0 && pty < h)
		{	
			result[common_hist_tab[p]*w*h+w*pty+ptx]+=1;
		}
	}
}

void GPUAdapter::testGPUSolution(cv::Rect box, Sample<float>&s)
{
	int blockSize = 32;

	s.x = 0;
	s.y = 0;

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(box.width/blockSize, box.height/blockSize);
    
    for(size_t t = 0; t < this->_treeAsTab.size(); ++t)
    {
		
		kernel<<<dimGrid, dimBlock>>>
		(this->resultGPU, _treeAsTab[t], 
		this->iWidth, this->iHeight, 
		this->w_integral, this->h_integral, 
		this->_features, this->_features_integral, 
		s, 
		this->lPXOff, this->lPYOff, 
		this->_common_hist_tab,  
		this->numLabels
		);
	}
	
}

void GPUAdapter::postKernel(cv::Mat*mapResult)
{
	cudaError_t ok;
	int size = this->iWidth*this->iHeight*this->numLabels*sizeof(int);
	ok=cudaMemcpy (this->result, this->resultGPU, size, cudaMemcpyDeviceToHost);
	if(ok != cudaSuccess)
	{
		std::cerr << "Cant get this->result back:"<<cudaGetErrorString(ok)<<"\n";
		exit(1);
	}
    int ptx, pty;
    size_t maxIdx;
    cv::Point pt;
    for (pty = 0; pty < this->iHeight; ++pty)
    for (ptx = 0; ptx < this->iWidth; ++ptx)
    {
        maxIdx = 0;

	
        for(int j = 1; j < this->numLabels; ++j)
        {
			if(this->result[j*this->iWidth*this->iHeight+pty*this->iWidth+ptx] > this->result[maxIdx*this->iWidth*this->iHeight+pty*this->iWidth+ptx])
				maxIdx = j;
            
			
        }
		pt.x = ptx;
		pt.y = pty;
        (*mapResult).at<uint8_t>(pt) = (uint8_t)maxIdx;
    }
	
	delete [] this->result;
	cudaFree(this->resultGPU);
	cudaFree(this->_features);
	cudaFree(this->_features_integral);
}


void GPUAdapter::destroy()
{
	for(int i=0; i < _treeAsTab.size();i++)
	{
		cudaFree(_treeAsTab[i]);
	}
	cudaFree(_common_hist_tab);
}
