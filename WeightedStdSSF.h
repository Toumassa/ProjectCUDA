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

#ifndef WEIGHTEDSTDSSF_H_
#define WEIGHTEDSTDSSF_H_

#include "SemanticSegmentationForests.h"

namespace vision
{

struct StdWErrorData
{
  double error;
};

class WeightedStdSSF: public AbstractSemanticSegmentationTree<StdWErrorData>
{
public:

  WeightedStdSSF(int seed = 0)
  {
    setRNG(seed);
  }
  virtual ~WeightedStdSSF()
  {
  }

protected:

  void initialize(const TNode<SplitData, Prediction> *node, StdWErrorData &errorData,
      Prediction &prediction) const
  {
    prediction.init(nClasses);
    const vector<LabelledSample<Sample, Label> > &samples = this->getLSamples();
   
    for (int i = node->getStart(); i < node->getEnd(); ++i)
    {
      ++prediction.hist[samples[i].label.value];
      ++prediction.n;
    }
    
    double norm = 0;
    for (int i = 0; i < nClasses; ++i)
    {
      prediction.p[i] = importance[i] * prediction.hist[i];
      norm += prediction.p[i];
    }

    errorData.error = 0;
    for (int i = 0; i < nClasses; ++i)
    {
      if (prediction.p[i] > 0)
      {
        prediction.p[i] /= (float)norm;
        errorData.error -= (float)prediction.p[i] * log(prediction.p[i]);
      }
    }
  }

  void updateError(StdWErrorData &newError, const StdWErrorData &errorData,
      const TNode<SplitData, Prediction> *node, Prediction &newLeft,
      Prediction &newRight) const
  {
    newError = errorData;
    newError.error = 0;
    
    double lError = 0, rError = 0, lNorm = 0, rNorm = 0;
    for (int j = 0; j < nClasses; ++j)
    {
      if(newLeft.p[j] > 0)
        newError.error -= importance[j] * newLeft.hist[j] * log(newLeft.p[j]);
      
      if(newRight.p[j] > 0)
        newError.error -= importance[j] * newRight.hist[j] * log(newRight.p[j]);
    }
  }

  double getError(const StdWErrorData &error) const
  {
    return error.error;
  }

};

}

#endif
