// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulò, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulò
//    October 2013
//
// =========================================================================================

#ifndef STRUCCLASSSSF_H_
#define STRUCCLASSSSF_H_

#include "SemanticSegmentationForests.h"


namespace vision
{

struct IErrorData
{
  IErrorData() : entropy(0) {lPos.resize(1);}
  IErrorData(size_t k) : entropy(0) {lPos.resize(k);}
  double entropy;
  vector<int> lPos;
};

template <class FeatureType> class StrucClassSSF : public AbstractSemanticSegmentationTree<IErrorData, FeatureType>
{
public:
  StrucClassSSF(int seed = 0) : xLDim(1), yLDim(1), numLPos(1)
  {
    this->setRNG(seed);

    // cout<<"Structured class tree with random boxes enabled"<<endl;
    this->bUseRandomBoxes = true;
  }
  virtual ~StrucClassSSF()
  {
  }
	
	
	/**
	 * Root Accessor
	 * 
	 **/
	 
	TNode<SplitData<FeatureType>, Prediction>* root()
	{
		return this->getRoot();
	}

  /***************************************************************************
   ***************************************************************************/

  vector<uint32_t>::const_iterator /* & */ predictPtr(Sample<FeatureType> &sample) const
  {
    TNode<SplitData<FeatureType>, Prediction> *curNode = this->getRoot();
    assert(curNode != NULL);
    SplitResult sr = SR_LEFT;
    while (!curNode->isLeaf() && sr != SR_INVALID)
    {
    if (this->bUseRandomBoxes==true)
        sr = this->split(curNode->getSplitData(), sample);
    else
        sr = AbstractSemanticSegmentationTree<IErrorData,FeatureType>::split(curNode->getSplitData(), sample);

    switch (sr)
      {
      case SR_LEFT:
        curNode = curNode->getLeft();
        break;
      case SR_RIGHT:
        curNode = curNode->getRight();
        break;
      default:
        break;
      }
    }

    return curNode->getPrediction().hist.begin();
  }

  /***************************************************************************
   ***************************************************************************/

  void setStrucClassParameters(int xDim, int yDim, int numPos)
  {
    if((xDim % 2) == 0 || (yDim % 2) == 0)
    {
      cout << "Label patch width/height must be odd numbers. Exiting!" << endl;
      exit(-1);
    }
    if(numPos < 1)
    {
      cout << "Joint probability dimension must be > 0. Exiting!" << endl;
      exit(-1);
    }

    xLDim = xDim;
    yLDim = yDim;
    lPXOff = xLDim / 2;
    lPYOff = yLDim / 2;
    numLPos = numPos;
  }

private:
  int lPXOff, lPYOff;
  int numLPos;
public:  
  int xLDim, yLDim;

  mutable vector<vector<float> > meanPatch;
  mutable vector<double> childPMF[2];

protected:

  /***************************************************************************
   ***************************************************************************/

  void initialize(const TNode<SplitData<FeatureType>, Prediction> *node, IErrorData &errorData,
      Prediction &prediction) const
  {
    meanPatch.resize(xLDim*yLDim, vector<float>(this->nClasses));

    childPMF[0].resize((int)pow((float)this->nClasses, (int)numLPos));
    childPMF[1].resize((int)pow((float)this->nClasses, (int)numLPos));
  }

  /***************************************************************************
   ***************************************************************************/

  void updateError(IErrorData &newError, const IErrorData &errorData,
      const TNode<SplitData<FeatureType>, Prediction> *node, Prediction &newLeft,
      Prediction &newRight) const
  {
    newError = IErrorData(numLPos);
    if (node->isLeaf())   // when its the first run decide for the new joint label positions
    {
      this->rng.fill(newError.lPos, RNG::UNIFORM, 0, xLDim * yLDim);
      newError.lPos[0] = (xLDim * yLDim) / 2;  // zero entry is always current center position
    }
    else
      newError.lPos = errorData.lPos;

    // reset variables
    fill(childPMF[0].begin(), childPMF[0].end(), 0);
    fill(childPMF[1].begin(), childPMF[1].end(), 0);
    double childNorm[2] = {0, 0};

    // auto sampleIt = this->samples.begin() + node->getStart();
    // auto sampleEnd = this->samples.begin() + node->getEnd();
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleIt = this->samples.begin() + node->getStart();
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleEnd = this->samples.begin() + node->getEnd();

    // get corresponding split results
    // auto splitResIt = this->splitResults.begin() + node->getStart();
    SplitResultsVector::const_iterator splitResIt = this->splitResults.begin() + node->getStart();
    size_t idx, centLab, tmpLab;
    Point labPos;
    Rect box(0, 0, 0, 0);

    for(; sampleIt != sampleEnd; ++sampleIt, ++splitResIt)
    {
      // get center label which is _always_ a valid one (ie.: centLab < nClasses)
      centLab = this->ts->getLabel(sampleIt->sample.imageId, sampleIt->sample.x, sampleIt->sample.y);
      idx = centLab;
      double incr = this->importance[centLab];
      box.width = this->ts->getImgWidth(sampleIt->sample.imageId);
      box.height = this->ts->getImgHeight(sampleIt->sample.imageId);

      for(size_t i = 1; i < newError.lPos.size(); ++i)
      {
        labPos.x = sampleIt->sample.x - lPXOff + newError.lPos[i] % xLDim;
        labPos.y = sampleIt->sample.y - lPYOff + newError.lPos[i] / xLDim;
        if(box.contains(labPos))
        {
          tmpLab = this->ts->getLabel(sampleIt->sample.imageId, labPos.x, labPos.y);
          if(tmpLab >= this->nClasses)
          {
            incr = 0;
            break;
          }
          idx = this->nClasses * idx + tmpLab;
          incr *= this->importance[tmpLab];
        }
        else
        {
          idx = this->nClasses * idx + centLab;
          incr *= this->importance[centLab];
        }
      }

      childPMF[*splitResIt][idx] += incr;
      childNorm[*splitResIt] += incr;
    }

    double val, partial;
    for(size_t child = 0; child < 2; ++child)
    {
      partial = 0;
      for(size_t i = 0; i < childPMF[child].size(); ++i)
      {
        val = childPMF[child][i] / childNorm[child];
        if(val > 0)
          partial -= val * log(val);
      }
      newError.entropy += childNorm[child] * partial;
    }
  }

  /***************************************************************************
   ***************************************************************************/

  double getError(const IErrorData &error) const
  {
    return error.entropy;
  }


  //////////////////////////////////////////////////////////////////////////////////
  //                                overrides
  //////////////////////////////////////////////////////////////////////////////////

  virtual bool updateLeafPrediction(const TNode<SplitData<FeatureType>, Prediction> *node, Prediction &newPrediction) const
  {
    // reset and initialize variables
    for(size_t i = 0; i < meanPatch.size(); ++i)
      fill(meanPatch[i].begin(), meanPatch[i].end(), 0);
    newPrediction.init((int)meanPatch.size());

    // auto sampleIt = this->samples.begin() + node->getStart();
    // auto sampleEnd = this->samples.begin() + node->getEnd();
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleIt = this->samples.begin() + node->getStart();
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleEnd = this->samples.begin() + node->getEnd();

    size_t centLab, labIdx, tmpLab;
    Rect box(0, 0, 0, 0);
    Point pt;

    // compute pixel-wise joint probability
    for(; sampleIt != sampleEnd; ++sampleIt)
    {
      centLab = this->ts->getLabel(sampleIt->sample.imageId, sampleIt->sample.x, sampleIt->sample.y);
      box.width = this->ts->getImgWidth(sampleIt->sample.imageId);
      box.height = this->ts->getImgHeight(sampleIt->sample.imageId);

      labIdx = 0;
      for(pt.y = sampleIt->sample.y - lPYOff; pt.y <= sampleIt->sample.y + lPYOff; ++pt.y)
        for(pt.x = sampleIt->sample.x - lPXOff; pt.x <= sampleIt->sample.x + lPXOff; ++pt.x, ++labIdx)
          if(box.contains(pt))
          {
            tmpLab = this->ts->getLabel(sampleIt->sample.imageId, pt.x, pt.y);
            if(tmpLab < this->nClasses)
              meanPatch[labIdx][tmpLab] += this->importance[tmpLab];
          }
          else
            meanPatch[labIdx][centLab] += this->importance[centLab];
    }

    // reset iterator
    // find sample that maximizes the joint-prob
    sampleIt = this->samples.begin() + node->getStart();

    // auto bestSample = sampleIt;
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator bestSample = sampleIt;
    float bestVal = -1, currVal;
    for(; sampleIt != sampleEnd; ++sampleIt)
    {
      centLab = this->ts->getLabel(sampleIt->sample.imageId, sampleIt->sample.x, sampleIt->sample.y);
      box.width = this->ts->getImgWidth(sampleIt->sample.imageId);
      box.height = this->ts->getImgHeight(sampleIt->sample.imageId);

      labIdx = 0;
      currVal = 0;
      for(pt.y = sampleIt->sample.y - lPYOff; pt.y <= sampleIt->sample.y + lPYOff; ++pt.y)
        for(pt.x = sampleIt->sample.x - lPXOff; pt.x <= sampleIt->sample.x + lPXOff; ++pt.x, ++labIdx)
          if(box.contains(pt))
          {
            tmpLab = this->ts->getLabel(sampleIt->sample.imageId, pt.x, pt.y);
            if(tmpLab < this->nClasses)
              currVal += meanPatch[labIdx][tmpLab];
          }
          else
            currVal += meanPatch[labIdx][centLab];

      if(currVal > bestVal)
      {
        bestSample = sampleIt;
        bestVal = currVal;
      }
    }

    // store prototype as appears in ground truth (may also contain labels >= nClasses)
    centLab = this->ts->getLabel(bestSample->sample.imageId, bestSample->sample.x, bestSample->sample.y);
    box.width = this->ts->getImgWidth(bestSample->sample.imageId);
    box.height = this->ts->getImgHeight(bestSample->sample.imageId);

    labIdx = 0;
    currVal = 0;
    for(pt.y = bestSample->sample.y - lPYOff; pt.y <= bestSample->sample.y + lPYOff; ++pt.y)
      for(pt.x = bestSample->sample.x - lPXOff; pt.x <= bestSample->sample.x + lPXOff; ++pt.x, ++labIdx)
        if(box.contains(pt))
          newPrediction.hist[labIdx] = this->ts->getLabel(bestSample->sample.imageId, pt.x, pt.y);
        else
          newPrediction.hist[labIdx] = (uint32_t)centLab;

    newPrediction.n = (uint32_t)(sampleEnd - (this->samples.begin() + node->getStart()));

    return true;
  }

  /***************************************************************************
   ***************************************************************************/

  virtual void updateLabelImportance(const vector<LabelledSample<Sample<FeatureType>, Label> > &samples)
  {
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleIt = this->samples.begin();
    typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::const_iterator sampleEnd = this->samples.end();

    this->importance.resize(this->nClasses);
    fill(this->importance.begin(), this->importance.end(), 0);
    for (; sampleIt != sampleEnd; ++sampleIt)
      ++(this->importance[sampleIt->label.value]);

    float norm = 0.f;
    for (uint16_t i = 0; i < this->nClasses; ++i)
      if(this->importance[i] > 0)
      {
        this->importance[i] = 1.0f / this->importance[i];
        norm += this->importance[i];
      }

    for (uint16_t i = 0; i < this->nClasses; ++i)
      this->importance[i] /= norm;
  }

  /***************************************************************************
   ***************************************************************************/

  virtual bool split(const TNode<SplitData<FeatureType>, Prediction> *node, SplitData<FeatureType> &splitData,
      Prediction &leftPrediction, Prediction &rightPrediction)
  {
    bool doSplit = node->getNSamples() >= (2 * this->minSamples) && node->getDepth() < this->maxDepth;

    if (doSplit)
    {
      // get randomly sampled split parameters
      splitData = this->generateSplit();

      // get current node samples
      typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::iterator sampleIt = this->samples.begin() + node->getStart();
      typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,IErrorData>::LSamplesVector::iterator sampleEnd = this->samples.begin() + node->getEnd();
      SplitResultsVector::iterator splitResIt = this->splitResults.begin() + node->getStart();

      FeatureType minVal = numeric_limits<FeatureType>::max();
      FeatureType maxVal = -numeric_limits<FeatureType>::max();

      // evaluate split on all samples and:
      // - all samples must be valid
      // - determine obtained test results

      for (; sampleIt != sampleEnd; ++sampleIt, ++splitResIt)
      {
    /*
#if USE_RANDOM_BOXES
        *splitResIt = this->split(splitData, sampleIt->sample);
#else
        *splitResIt = AbstractSemanticSegmentationTree<IErrorData,FeatureType>::split(splitData, sampleIt->sample);
#endif
*/
        if (this->bUseRandomBoxes==true)
            *splitResIt = this->split(splitData, sampleIt->sample);
        else
            *splitResIt = AbstractSemanticSegmentationTree<IErrorData,FeatureType>::split(splitData, sampleIt->sample);

        // check response intervals boundaries
        if (*splitResIt != SR_INVALID)
        {
          if (sampleIt->sample.value < minVal)
            minVal = sampleIt->sample.value;
          if (sampleIt->sample.value > maxVal)
            maxVal = sampleIt->sample.value;
        }
        else
          cout << "0: Error in split-override of StrucClassSSF. All samples _have_ to be valid!" << endl;
      }

      // determine sample threshold from previously defined ranges
      splitData.thres = (FeatureType) this->rng.uniform(minVal, maxVal);

      // child counters
      uint32_t numValid[2] = {0, 0};

      // reset sample and splitresult iterators to start
      sampleIt = this->samples.begin() + node->getStart();
      splitResIt = this->splitResults.begin() + node->getStart();

      for (; sampleIt != sampleEnd; ++sampleIt, ++splitResIt)
      {
        // we need to set now if a sample goes left or right according to splitData.thres
        if (*splitResIt != SR_INVALID)
        {
          if (sampleIt->sample.value < splitData.thres)
          {
            *splitResIt = SR_LEFT;
            ++numValid[SR_LEFT];
          }
          else
          {
            *splitResIt = SR_RIGHT;
            ++numValid[SR_RIGHT];
          }
        }
        else
          cout << "1: Error in split-override of StrucClassSSF. All samples _have_ to be valid!" << endl;
      }

      // final check: if both childs hold sufficiently many samples
      doSplit = numValid[SR_LEFT] >= this->minSamples && numValid[SR_RIGHT] >= this->minSamples;
    }

    return doSplit;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #if USE_RANDOM_BOXES
  // begin random probe box splits modification
  // split function using the randomly selected box parameters
  SplitResult split(const SplitData<FeatureType> &splitData, Sample<FeatureType> &sample) const
  {
    sample.value = this->ts->getValue(sample.imageId, splitData.channel0, sample.x, sample.y); //, sample.x+1, sample.y+1);
    SplitResult centerResult = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;

#ifdef VERBOSE_PREDICTION
    std::cerr << "CPU-split: x=" << sample.x << " y=" << sample.y << " val=" << sample.value << " centerResult=" << (centerResult==SR_LEFT ? "L" : "R") << " fType=" << (int)  splitData.fType;
#endif

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
                       - this->ts->getValueIntegral(sample.imageId, splitData.channel1, pt3.x, pt3.y, pt4.x, pt4.y) / norm2;
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

  /***************************************************************************
   ***************************************************************************/

  void write(const SplitData<FeatureType> &splitData, ostream &out) const
  {
    out << splitData.dx1 << " " << splitData.dy1 << " ";
    out << (uint32_t) (splitData.bw1) << " " << (uint32_t) (splitData.bh1) << " ";
    out << splitData.dx2 << " " << splitData.dy2 << " ";
    out << (uint32_t) (splitData.bw2) << " " << (uint32_t) (splitData.bh2) << " ";
    out << (uint32_t) (splitData.channel0) << " " << (uint32_t) (splitData.channel1) << " "
        << (uint32_t) (splitData.fType) << " " << splitData.thres;
  }

  /***************************************************************************
   ***************************************************************************/

  void read(SplitData<FeatureType> &splitData, istream &in) const
  {
    in >> splitData.dx1 >> splitData.dy1;
    uint32_t tmp;
    in >> tmp;
    splitData.bw1 = (int8_t) tmp;
    in >> tmp;
    splitData.bh1 = (int8_t) tmp;

    in >> splitData.dx2 >> splitData.dy2;
    in >> tmp;
    splitData.bw2 = (int8_t) tmp;
    in >> tmp;
    splitData.bh2 = (int8_t) tmp;

    in >> tmp;
    splitData.channel0 = (uint8_t) tmp;
    in >> tmp;
    splitData.channel1 = (uint8_t) tmp;
    in >> tmp;
    splitData.fType = (uint8_t) tmp;
    in >> splitData.thres;
  }
  // end random probe box splits modification
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// #endif

  virtual void writeHeader(ostream &out) const
  {
    out << this->nClasses << " " << xLDim << " " << yLDim << endl;
    for(size_t i = 0; i < this->importance.size(); ++i)
      out << this->importance[i] << " ";
  }

  /***************************************************************************
   ***************************************************************************/

  virtual void readHeader(istream &in)
  {
    in >> this->nClasses;
    in >> xLDim;
    in >> yLDim;

    lPXOff = xLDim / 2;
    lPYOff = yLDim / 2;

    this->importance.resize(this->nClasses);
    for(size_t i = 0; i < this->nClasses; ++i)
      in >> this->importance[i];
  }

  /***************************************************************************
   ***************************************************************************/


  virtual void write(const TNode<SplitData<FeatureType>, Prediction> *node, ostream &out) const
  {
    out << (node->isLeaf() ? "L " : "N ");
    out << node->getStart() << " " << node->getEnd() << " " << node->getDepth() << " ";
    if(!node->isLeaf())
    {
    /*
#if USE_RANDOM_BOXES
      write(node->getSplitData(), out);
#else
      AbstractSemanticSegmentationTree<IErrorData, FeatureType>::write(node->getSplitData(), out);
#endif
*/
    if (this->bUseRandomBoxes==true)
      write(node->getSplitData(), out);
    else
      AbstractSemanticSegmentationTree<IErrorData, FeatureType>::write(node->getSplitData(), out);

    }
    else
      write(node->getPrediction(), out);

    // debug of samples location
    //if(node->isLeaf())
    //{
    //  const LSamplesVector &samples = this->getLSamples();
    //  LSamplesVector::const_iterator sampleIt = samples.begin() + node->getStart();
    //  const LSamplesVector::const_iterator sampleEnd = samples.begin() + node->getEnd();

    //  for(; sampleIt != sampleEnd; ++sampleIt)
    //    out << sampleIt->sample.imageId << " " << sampleIt->sample.x << " " << sampleIt->sample.y << " ";
    //}

    out << endl;
    if (!node->isLeaf())
    {
      write(node->getLeft(), out);
      write(node->getRight(), out);
    }
  }

  /***************************************************************************
   ***************************************************************************/

  virtual void read(TNode<SplitData<FeatureType>, Prediction> *node, istream &in) const
  {
    char type;
    in >> type;
    bool isLeaf = type == 'L';

    if (type!='L' && type!='N')
    {
        cout<<"ERROR: unknown node type: "<<type<<endl;
        exit(-1);
    }
    int start, end, depth;
    in >> start;
    in >> end;
    in >> depth;
    node->setStart(start);
    node->setEnd(end);
    node->setDepth(depth);

    if(!isLeaf)
    {
      SplitData<FeatureType> splitData;
      /*
#if USE_RANDOM_BOXES
      read(splitData, in);
#else
      AbstractSemanticSegmentationTree<IErrorData,FeatureType>::read(splitData, in);
#endif
    */
    if (this->bUseRandomBoxes==true)
      read(splitData, in);
    else
      AbstractSemanticSegmentationTree<IErrorData,FeatureType>::read(splitData, in);

    // cout<<splitData<<endl;

      node->setSplitData(splitData);
    }
    else
    {
      Prediction prediction;
      read(prediction, in);
      node->setPrediction(prediction);
    }
    if (!isLeaf)
    {
      node->split(node->getStart(), node->getStart());
      read(node->getLeft(), in);
      read(node->getRight(), in);
    }
  }

  /***************************************************************************
   ***************************************************************************/

  void write(const Prediction &prediction, ostream &out) const
  {
    for (int i = 0; i < prediction.hist.size(); ++i)
      out << prediction.hist[i] << " ";
    out << prediction.n << " ";
  }

  /***************************************************************************
   ***************************************************************************/

  void read(Prediction &prediction, istream &in) const
  {
    prediction.init(xLDim * yLDim);
    for (int i = 0; i < prediction.hist.size(); ++i)
      in >> prediction.hist[i];
    in >> prediction.n;
  }
};

}

#endif
