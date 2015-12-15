#include "GPUAdapter.h"
#include "RandomForest.h"




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
SplitResult split(SplitData<float> splitData, Sample<float> &sample)
{
	 sample.value = this->ts->getValue(sample.imageId, splitData.channel0, sample.x, sample.y); //, sample.x+1, sample.y+1);
     SplitResult centerResult = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;

    if (splitData.fType == 0) // single probe (center only)
    {
#ifdef VERBOSE_PREDICTION
      std::cerr << "\n";
#endif
      return centerResult;
    }

    // for cases when we have non-centered probe types
    Point pt1, pt2, pt3, pt4;

    pt1.x = sample.x + splitData.dx1 - splitData.bw1;
    pt1.y = sample.y + splitData.dy1 - splitData.bh1;

    pt2.x = sample.x + splitData.dx1 + splitData.bw1 + 1; // remember -> integral images have size w+1 x h+1
    pt2.y = sample.y + splitData.dy1 + splitData.bh1 + 1;

    int16_t w = this->ts->getImgWidth(sample.imageId);
    int16_t h = this->ts->getImgHeight(sample.imageId);

#ifdef VERBOSE_PREDICTION
    std::cerr << " pt1=" << pt1 << " pt2=" << pt2;
#endif

    if (pt1.x < 0 || pt2.x < 0 || pt1.y < 0 || pt2.y < 0 ||
        pt1.x > w || pt2.x > w || pt1.y > h || pt2.y > h) // due to size correction in getImgXXX we dont have to check \geq
    {
#ifdef VERBOSE_PREDICTION
      std::cerr << "\n";
#endif
      return centerResult;
    }
    else
    {
      if (splitData.fType == 1) // single probe (center - offset)
      {
        int16_t norm1 = (pt2.x - pt1.x) * (pt2.y - pt1.y);
        sample.value -= this->ts->getValueIntegral(sample.imageId, splitData.channel0, pt1.x, pt1.y, pt2.x, pt2.y) / norm1;
#ifdef VERBOSE_PREDICTION
      	std::cerr << "new-val1= " << sample.value;
#endif
      }
      else                      // pixel pair probe test
      {
        pt3.x = sample.x + splitData.dx2 - splitData.bw2;
        pt3.y = sample.y + splitData.dy2 - splitData.bh2;

        pt4.x = sample.x + splitData.dx2 + splitData.bw2 + 1;
        pt4.y = sample.y + splitData.dy2 + splitData.bh2 + 1;

#ifdef VERBOSE_PREDICTION
    	std::cerr << " pt3=" << pt3 << " pt4=" << pt4;
#endif

        if (pt3.x < 0 || pt4.x < 0 || pt3.y < 0 || pt4.y < 0 ||
            pt3.x > w || pt4.x > w || pt3.y > h || pt4.y > h)
        {
#ifdef VERBOSE_PREDICTION
      	   std::cerr << "\n";
#endif
          return centerResult;
        }

        int16_t norm1 = (pt2.x - pt1.x) * (pt2.y - pt1.y);
        int16_t norm2 = (pt4.x - pt3.x) * (pt4.y - pt3.y);

        if (splitData.fType == 2)    // sum of pair probes
        {
          sample.value = this->ts->getValueIntegral(sample.imageId, splitData.channel0, pt1.x, pt1.y, pt2.x, pt2.y) / norm1
                       + this->ts->getValueIntegral(sample.imageId, splitData.channel1, pt3.x, pt3.y, pt4.x, pt4.y) / norm2;
        }
        else if (splitData.fType == 3)  // difference of pair probes
        {
          sample.value = this->ts->getValueIntegral(sample.imageId, splitData.channel0, pt1.x, pt1.y, pt2.x, pt2.y) / norm1
                       - this->ts->getValueIntegral(sample.imageId, 
                        .channel1, pt3.x, pt3.y, pt4.x, pt4.y) / norm2;
        }
        else
          cout << "ERROR: Impossible case in splitData in StrucClassSSF::split(...)"
               << endl;

#ifdef VERBOSE_PREDICTION
      	std::cerr << " new-val23= " << sample.value;
#endif

      }
    }

    SplitResult res = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;
#ifdef VERBOSE_PREDICTION
      	   std::cerr << " result=" << (res==SR_LEFT ? "L" : "R") << "\n";
#endif
    return res;
}

__device__
void predict(int *returnStartHistTab, int *returnCountHistTab, ANode* tree, float* features, float* features_integral, Sample<float> &sample)
{
	int curNode = 0; //initialising to Root

    SplitResult sr = SR_LEFT;
    while ((tree[curNode].left != -1) && sr != SR_INVALID)
    {
    /*if (this->bUseRandomBoxes==true)*/
    
    sr = this->split(tree[curNode].splitData, sample);

    /*else
        sr = AbstractSemanticSegmentationTree<IErrorData,FeatureType>::split(curNode->getSplitData(), sample);*/

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
    (*returnCountHistTab) = tree[curNode].common_hist_tab_size;
}