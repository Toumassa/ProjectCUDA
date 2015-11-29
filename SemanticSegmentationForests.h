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

#ifndef SEMANTICSEGMENTATIONFORESTS_H_
#define SEMANTICSEGMENTATIONFORESTS_H_

#include "Global.h"
#include <vector>
#include "RandomForest.h"
#include "ImageData.h"
#include "TrainingSet.h"
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <opencv/cv.h>
#include <limits>

// #define USE_RANDOM_BOXES 1

using namespace std;
using namespace cv;

namespace vision
{

template <typename FeatureType> struct Sample
{
  int16_t x, y;
  uint16_t imageId;
  FeatureType value;
};

struct Label
{
  // center label (as usual)
  uint8_t value;
};

struct Prediction
{
  Prediction() :
      n(0)
  {
  }

  Prediction(int k) :
      n(0)
  {
    hist.resize(k);
    p.resize(k);
  }

  void init(int k)
  {
    if (hist.size() != k)
    {
      hist.resize(k);
    }
    if (p.size() != k)
    {
      p.resize(k);
    }
    for (int i = 0; i < k; ++i)
    {
      hist[i] = 0;
      p[i] = 0;
    }
    n = 0;
  }

  vector<uint32_t> hist;
  uint32_t n;
  vector<float> p;
};

template<typename FeatureType> struct SplitData
{
  SplitData() : dx1(0), dx2(0), dy1(0), dy2(0), channel0(0), channel1(0), fType(0), thres(0)
  {}

  int16_t dx1, dx2;
  int16_t dy1, dy2;
  int8_t bw1, bh1, bw2, bh2;
  uint8_t channel0;    // number of feature channels is restricted by 255
  uint8_t channel1;
  uint8_t fType;           // CW: split type
  FeatureType thres;       // short is sufficient in range
};

template<class ErrorData, typename FeatureType>
class AbstractSemanticSegmentationTree: public RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,ErrorData>
{
  public:
    bool bUseRandomBoxes;


public:
  AbstractSemanticSegmentationTree() :
      nClasses(0), ts(NULL), minSamples(minSamples), nTrials(500), maxDepth(0), samplingType(
          HOMOGENEOUS), interleaved(false), samplingParam(1000), maxProbeOffset(30)
  {
    bUseRandomBoxes = false;
  }
  virtual ~AbstractSemanticSegmentationTree()
  {
  }

  void setTrainingSet(const TrainingSet<FeatureType> *pt)
  {
    this->ts = pt;
    nClasses = ts->getNLabels();
    importance.resize(nClasses, 1.0);
  }

  void setMinSamples(uint32_t minSamples)
  {
    this->minSamples = minSamples;
  }

  void setNTests(uint32_t nTrials)
  {
    this->nTrials = nTrials;
  }

  void setMaxDepth(uint16_t maxDepth)
  {
    this->maxDepth = maxDepth;
  }

  void setSamplingType(SAMPLING_TYPE st)
  {
    this->samplingType = st;
  }

  void setSamplingParam(uint16_t samplingParam)
  {
    this->samplingParam = samplingParam;
  }

  void setTrainingType(bool interleaved)
  {
    this->interleaved = interleaved;
  }

  void setMaxProbeOffset(uint16_t offset)
  {
    this->maxProbeOffset = offset;
  }

  void setRNG(int seed)
  {
    this->rng = RNG(seed);
  }

  vector<float> getLabelImportance() const
  {
    return importance;
  }

  void train(uint16_t minSamples, uint16_t maxDepth, uint32_t nTrials, SAMPLING_TYPE samplingType,
      uint32_t samplingParam, uint16_t maxProbeOffset, bool trainInterleaved, bool useWeights =
          false, bool getLabelStats = false)
  {
    if (nClasses == 0 || ts == NULL)
    {
      cout << "Need to provide training data first (call setTrainingSet(...))" << endl;
      return;
    }

    this->minSamples = minSamples;
    this->maxDepth = maxDepth;
    this->nTrials = nTrials;
    this->samplingType = samplingType;
    this->samplingParam = samplingParam;
    this->maxProbeOffset = maxProbeOffset;
    this->interleaved = trainInterleaved;

    vector<LabelledSample<Sample<FeatureType>, Label> > samples; // = this->generateSamples(getLabelStats);

    generateSamplesEven(getLabelStats, samples);

    vector<unsigned int> vectNbSamplesPerLabel;
    unsigned int iSample, iLabel;

    vectNbSamplesPerLabel.resize(nClasses);
    for (iLabel=0; iLabel<nClasses; iLabel++)
        vectNbSamplesPerLabel[iLabel] = 0;

    for (iSample=0; iSample<samples.size(); iSample++)
        vectNbSamplesPerLabel[samples[iSample].label.value]++;

    for (iLabel=0; iLabel<nClasses; iLabel++)
        cout<<iLabel<<": "<<vectNbSamplesPerLabel[iLabel]<<endl;

    if (useWeights)
      updateLabelImportance(samples);

    if (bUseRandomBoxes==true)
        cout<<"Random boxes enabled"<<endl;
    else
        cout<<"Random boxes disabled"<<endl;

    RandomTree<SplitData<FeatureType>, Sample<FeatureType>, Label, Prediction, ErrorData>::train(samples, nTrials,
        interleaved);
  }

protected:

  virtual bool split(const TNode<SplitData<FeatureType>, Prediction> *node, SplitData<FeatureType> &splitData,
      Prediction &leftPrediction, Prediction &rightPrediction)
  {
    bool doSplit = node->getNSamples() >= (2 * minSamples) && node->getDepth() < maxDepth;

    if (doSplit)
    {
      // get randomly sampled split parameters
      splitData = generateSplit();

      // get current node samples
      // vector<LabelledSample<Sample<FeatureType>, Label> >
      // auto sampleIt = this->samples.begin() + node->getStart();
      // auto sampleEnd = this->samples.begin() + node->getEnd();
      typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,ErrorData>::LSamplesVector::iterator sampleIt = this->samples.begin() + node->getStart();
      typename RandomTree<SplitData<FeatureType>,Sample<FeatureType>,Label,Prediction,ErrorData>::LSamplesVector::iterator sampleEnd = this->samples.begin() + node->getEnd();

      // get corresponding split results
      // SplitResultsVector &splitResults = this->getSplitResults();
      SplitResultsVector::iterator splitResIt = this->splitResults.begin() + node->getStart();

      FeatureType minVal = numeric_limits<FeatureType>::max();
      FeatureType maxVal = -numeric_limits<FeatureType>::max();

      // evaluate split on all samples and:
      // - determine if sample is valid
      // - determine obtained test results
      for (; sampleIt != sampleEnd; ++sampleIt, ++splitResIt)
      {
        *splitResIt = split(splitData, sampleIt->sample);

        // check response intervals boundaries
        if (*splitResIt != SR_INVALID)
        {
          if (sampleIt->sample.value < minVal)
            minVal = sampleIt->sample.value;
          if (sampleIt->sample.value > maxVal)
            maxVal = sampleIt->sample.value;
        }
      }

      // determine sample threshold from previously defined ranges
      splitData.thres = (FeatureType) rng.uniform(minVal, maxVal);

      // reset child predictions
      leftPrediction.init(nClasses);
      rightPrediction.init(nClasses);
      vector<uint32_t> numValid(2, 0);

      // reset sample and splitresult iterators to start
      sampleIt = this->samples.begin() + node->getStart();
      splitResIt = this->splitResults.begin() + node->getStart();
      float normLeft = 0, normRight = 0;

      for (; sampleIt != sampleEnd; ++sampleIt, ++splitResIt)
      {
        // only the SR_INVALID flag can be trusted so far
        // we need to set now if a sample goes left or right according to splitData.thres
        if (*splitResIt != SR_INVALID)
        {
          if (sampleIt->sample.value < splitData.thres)
          {
            *splitResIt = SR_LEFT;
            ++numValid[SR_LEFT];
            ++leftPrediction.hist[sampleIt->label.value];
            ++leftPrediction.n;
            leftPrediction.p[sampleIt->label.value] += importance[sampleIt->label.value];
            normLeft += importance[sampleIt->label.value];
          }
          else
          {
            *splitResIt = SR_RIGHT;
            ++numValid[SR_RIGHT];
            ++rightPrediction.hist[sampleIt->label.value];
            ++rightPrediction.n;
            rightPrediction.p[sampleIt->label.value] += importance[sampleIt->label.value];
            normRight += importance[sampleIt->label.value];
          }
        }
      }

      // final check: if both childs hold sufficiently many samples
      doSplit = numValid[SR_LEFT] >= minSamples && numValid[SR_RIGHT] >= minSamples;
      if (doSplit)
      {
        // normalize
        for (int i = 0; i < nClasses; ++i)
        {
          leftPrediction.p[i] /= normLeft;
          rightPrediction.p[i] /= normRight;
        }
      }

    }
    return doSplit;
  }

  SplitResult split(const SplitData<FeatureType> &splitData, Sample<FeatureType> &sample) const
  {
    sample.value = ts->getValue(sample.imageId, splitData.channel0, sample.x, sample.y);
    SplitResult centerResult = (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;

    if (splitData.fType == 0) // single probe (center only)
    {
      return centerResult;
    }

    int16_t x1, y1, x2, y2;

    x1 = sample.x + splitData.dx1;
    y1 = sample.y + splitData.dy1;
    if (x1 < 0 || y1 < 0 || x1 >= ts->getImgWidth(sample.imageId)
        || y1 >= ts->getImgHeight(sample.imageId))
    {
      return centerResult;
    }
    else
    {
      if (splitData.fType == 1) // single probe (center - offset)
      {
        sample.value = ts->getValue(sample.imageId, splitData.channel0, sample.x, sample.y)
            - ts->getValue(sample.imageId, splitData.channel0, x1, y1);
      }
      else                      // pixel pair probe test
      {
        x2 = sample.x + splitData.dx2;
        y2 = sample.y + splitData.dy2;
        if (x2 < 0 || y2 < 0 || x2 >= ts->getImgWidth(sample.imageId)
            || y2 >= ts->getImgHeight(sample.imageId))
        {
          return centerResult;
        }

        if (splitData.fType == 2)    // sum of pair probes
        {
          sample.value = ts->getValue(sample.imageId, splitData.channel0, x1, y1)
              + ts->getValue(sample.imageId, splitData.channel1, x2, y2);
        }
        else if (splitData.fType == 3)  // difference of pair probes
        {
          sample.value = ts->getValue(sample.imageId, splitData.channel0, x1, y1)
              - ts->getValue(sample.imageId, splitData.channel1, x2, y2);
        }
        else
          cout << "ERROR: Impossible case in splitData in SemanticSegmentationForest::split(...)"
              << endl;
      }
    }

    return (sample.value < splitData.thres) ? SR_LEFT : SR_RIGHT;
  }

  SplitData<FeatureType> generateSplit()
  {
    SplitData<FeatureType> sd;

    // generate random splits
    sd.channel0 = (uint8_t) rng.uniform(0, (int) ts->getNChannels());
    sd.channel1 = (uint8_t) rng.uniform(0, (int) ts->getNChannels());
    sd.fType = (uint8_t) rng.uniform(0, 4);
    //sd.fType = (uint8_t) rng.uniform(2, 4); // pair probes only
    sd.dx1 = rng.uniform(-maxProbeOffset + 1, maxProbeOffset);
    sd.dx2 = rng.uniform(-maxProbeOffset + 1, maxProbeOffset);
    sd.dy1 = rng.uniform(-maxProbeOffset + 1, maxProbeOffset);
    sd.dy2 = rng.uniform(-maxProbeOffset + 1, maxProbeOffset);

// #if USE_RANDOM_BOXES
    if (bUseRandomBoxes==true)
    {
        sd.bw1 = (int8_t)rng.uniform(0, maxProbeOffset);
        sd.bh1 = (int8_t)rng.uniform(0, maxProbeOffset);
        sd.bw2 = (int8_t)rng.uniform(0, maxProbeOffset);
        sd.bh2 = (int8_t)rng.uniform(0, maxProbeOffset);
    }
// #endif

    sd.thres = (FeatureType)0;

    return sd;
  }

  void writeHeader(ostream &out) const
  {
    out << nClasses;
  }

  void readHeader(istream &in)
  {
    in >> nClasses;
  }

  void write(const Sample<FeatureType> &sample, ostream &out) const
  {
    out << sample.x << " " << sample.y << " " << sample.imageId;
  }

  void read(Sample<FeatureType> &sample, istream &in) const
  {
    in >> sample.x >> sample.y >> sample.imageId;
  }

  void write(const Prediction &prediction, ostream &out) const
  {
    for (int i = 0; i < nClasses; ++i)
      out << prediction.hist[i] << " ";
    out << prediction.n << " ";
    for (int i = 0; i < nClasses; ++i)
      out << prediction.p[i] << " ";
  }

  void read(Prediction &prediction, istream &in) const
  {
    prediction.init(nClasses);
    for (int i = 0; i < nClasses; ++i)
      in >> prediction.hist[i];
    in >> prediction.n;
    for (int i = 0; i < nClasses; ++i)
      in >> prediction.p[i];
  }

  void write(const Label &label, ostream &out) const
  {
    out << label.value;
  }

  void read(Label &label, istream &in) const
  {
    in >> label.value;
  }

  void write(const SplitData<FeatureType> &splitData, ostream &out) const
  {
    out << splitData.dx1 << " " << splitData.dy1 << " ";
    out << splitData.dx2 << " " << splitData.dy2 << " ";
    out << (uint32_t) (splitData.channel0) << " " << (uint32_t) (splitData.channel1) << " "
        << (uint32_t) (splitData.fType) << " ";

    // splitData.thres;
    if (splitData.thres == numeric_limits<FeatureType>::infinity())
    {
        out << numeric_limits<FeatureType>::max();
        cout<<"WARNING in AbstractSemanticSegmentationTree::write(SpliData...) : threshold is infinite"<<endl;
        // exit(-1);
    }
    else
        out << splitData.thres;
  }

  void read(SplitData<FeatureType> &splitData, istream &in) const
  {
    in >> splitData.dx1 >> splitData.dy1;
    in >> splitData.dx2 >> splitData.dy2;
    uint32_t tmp;
    in >> tmp;
    splitData.channel0 = (uint8_t) tmp;
    in >> tmp;
    splitData.channel1 = (uint8_t) tmp;
    in >> tmp;
    splitData.fType = (uint8_t) tmp;
    in >> splitData.thres;
  }

private:

  vector<LabelledSample<Sample<FeatureType>, Label> > generateSamples(bool computeLabelStats)
  {
    vector<LabelledSample<Sample<FeatureType>, Label> > samples;
    samples.clear();

    for (uint16_t i = 0; i < ts->getNbImages(); ++i)
    {
        // cout<<"Geberating samples on image "<<i<<endl;
      cv::Mat labImg = ts->getLabelImage(i);
      int w = labImg.cols, h = labImg.rows;

      if (samplingType == HOMOGENEOUS)
      {
        // cout<<"HOMOGENEOUS sampling"<<endl;
        samples.reserve(samples.size() + nClasses * samplingParam); // not optimal, but OK
        vector<vector<uint32_t> > pixLabels(nClasses, vector<uint32_t>());
        MatConstIterator_<uint8_t> labIt = labImg.begin<uint8_t>(), labEnd = labImg.end<uint8_t>();
        uint32_t pixIdx = 0;
        for (; labIt != labEnd; ++labIt, ++pixIdx)
          if (*labIt < nClasses) // consider only classes with label < nClasses
            pixLabels[*labIt].push_back(pixIdx);

        for (uint16_t cl = 0; cl < nClasses; ++cl)
        {
          if (pixLabels[cl].size() > samplingParam)
            randShuffle(Mat(pixLabels[cl]), 3, &rng);

          for (uint32_t smpl = 0; smpl < MIN(samplingParam, pixLabels[cl].size()); ++smpl)
          {
            LabelledSample<Sample<FeatureType>, Label> s;
            s.sample.imageId = i;
            s.sample.x = (int16_t) (pixLabels[cl][smpl] % w);
            s.sample.y = (int16_t) (pixLabels[cl][smpl] / w);
            s.label.value = (uint8_t) cl;

            samples.push_back(s);
          }
        }
      }
      else if (samplingType == STRIDE)
      {
        // cout<<"STRIDE sampling"<<endl;
        samples.reserve(
            samples.size()
                + (uint32_t) (1 + (w - 1) / samplingParam) * (1 + (h - 1) / samplingParam));

        int randOff = samplingParam / 2;
        int posx, posy;

        for (int y = 0; y < h; y += samplingParam)
        {
          for (int x = 0; x < w; x += samplingParam)
          {
            posx = x + rng.uniform(-randOff, randOff + 1);
            posy = y + rng.uniform(-randOff, randOff + 1);

            // if sample is out of bounds or its class is not of interest to us, we simply skip it
            if (posx < 0 || posy < 0 || posx >= w || posy >= h
                || labImg.at<uint8_t>(posy, posx) >= nClasses)
              continue;

            LabelledSample<Sample<FeatureType>, Label> s;
            s.sample.imageId = i;
            s.sample.x = (int16_t) posx;
            s.sample.y = (int16_t) posy;
            s.label.value = labImg.at<uint8_t>(posy, posx);

            samples.push_back(s);
          }
        }
      }
      else
        cout << "Invalid sampling type in SemanticSegmentationForest::generateSamples()" << endl;


    }
#pragma omp parallel
    {
#pragma omp critical
    printDataDistribution(samples);
    }
    return samples;
  }

    void generateSamplesEven(bool computeLabelStats, vector<LabelledSample<Sample<FeatureType>, Label> > &samples)
    {
        uint16_t iImg;
        uint8_t iLabel, iLabelWanted;
        uint8_t *pPixel;
        int iWidth, iHeight;
        cv::Mat labImg;
        cv::Point pt;
        vector<unsigned int> vectNbSamplesPerLabel, vectNbSamplesPerLabelWanted;
        vector<vector<unsigned int> > vectNbSamplesPerImageLabel, vectNbSamplesPerImageLabelWanted;
        unsigned int iNbSamplesMin, iNbSamplesWanted;

        samples.clear();
        vectNbSamplesPerLabel.resize(nClasses);
        vectNbSamplesPerLabelWanted.resize(nClasses);

        vectNbSamplesPerImageLabel.resize(ts->getNbImages());
        vectNbSamplesPerImageLabelWanted.resize(ts->getNbImages());

        for (iLabel=0; iLabel<nClasses; iLabel++)
            vectNbSamplesPerLabel[iLabel] = 0;

        // Count total number of samples per label and per image
        for (iImg = 0; iImg < ts->getNbImages(); iImg++)
        {
            labImg = ts->getLabelImage(iImg);
            iWidth = labImg.cols;
            iHeight = labImg.rows;

            vectNbSamplesPerImageLabel[iImg].resize(nClasses);
            vectNbSamplesPerImageLabelWanted[iImg].resize(nClasses);
            for (iLabel=0; iLabel<nClasses; iLabel++)
                vectNbSamplesPerImageLabel[iImg][iLabel] = 0;

            for (pt.y=0; pt.y<iHeight; pt.y++)
            {
                for (pt.x=0, pPixel = labImg.ptr(pt.y); pt.x<iWidth; pt.x++, pPixel++)
                {
                    iLabel = *pPixel;
                    vectNbSamplesPerImageLabel[iImg][iLabel]++;
                    vectNbSamplesPerLabel[iLabel]++;
                }
            }
        }

        // Find the minimal amount of samples
        iNbSamplesMin = INT_MAX;
        for (iLabel=0; iLabel<nClasses; iLabel++)
        {
            if (vectNbSamplesPerLabel[iLabel]!=0)
            {
                if (vectNbSamplesPerLabel[iLabel]<iNbSamplesMin)
                    iNbSamplesMin = vectNbSamplesPerLabel[iLabel];
            }
        }

        // Set wanted number of samples per class
        iNbSamplesWanted = 0;
        for (iLabel=0; iLabel<nClasses; iLabel++)
        {
            if (vectNbSamplesPerLabel[iLabel]==0)
            {
                vectNbSamplesPerLabelWanted[iLabel] = 0;
                for (iImg=0; iImg<ts->getNbImages(); iImg++)
                    vectNbSamplesPerImageLabelWanted[iImg][iLabel] = 0;
            }
            else {
                // unsigned int iNbMin = 2000;
                vectNbSamplesPerLabelWanted[iLabel] = min(/*iNbMin,*/ vectNbSamplesPerLabel[iLabel], 2*iNbSamplesMin);
                iNbSamplesWanted += vectNbSamplesPerLabelWanted[iLabel];

                cout<<(int)iLabel<<": nb="<<vectNbSamplesPerLabel[iLabel]
                    <<"  nb wanted="<<vectNbSamplesPerLabelWanted[iLabel]<<endl;

                // Set wanted number of samples per class and per image
                for (iImg=0; iImg<ts->getNbImages(); iImg++)
                {
                    vectNbSamplesPerImageLabelWanted[iImg][iLabel] = (unsigned int)(
                        (float)vectNbSamplesPerImageLabel[iImg][iLabel] * (float)vectNbSamplesPerLabelWanted[iLabel]
                        / (float)vectNbSamplesPerLabel[iLabel]);
                }
            }
        }

        /*
        cout<<"CHECK"<<endl;
        for (iLabel=0; iLabel<nClasses; iLabel++)
        {
            unsigned int iNbSamplesWantedCheck = 0;
            for (iImg=0; iImg<ts->getNbImages(); iImg++)
                iNbSamplesWantedCheck += vectNbSamplesPerImageLabelWanted[iImg][iLabel];
            cout<<iLabel<<": "<<iNbSamplesWantedCheck<<endl;
        }*/

        samples.reserve(iNbSamplesWanted);
        // cout<<"Nb samples wanted = "<<iNbSamplesWanted<<endl;

        for (iImg=0; iImg<ts->getNbImages(); iImg++)
        {
            labImg = ts->getLabelImage(iImg);

            for (iLabelWanted=0; iLabelWanted<nClasses; iLabelWanted++)
            {
                if (vectNbSamplesPerImageLabelWanted[iImg][iLabelWanted]!=0)
                {
                    LabelledSample<Sample<FeatureType>, Label> s;
                    s.sample.imageId = iImg;
                    s.label.value = iLabelWanted;

                    if (vectNbSamplesPerImageLabel[iImg][iLabelWanted]==vectNbSamplesPerImageLabelWanted[iImg][iLabelWanted])
                    {
                        // Take all pixels of current label
                        for (pt.y=0; pt.y<iHeight; pt.y++)
                        {
                            for (pt.x=0, pPixel = labImg.ptr(pt.y); pt.x<iWidth; pt.x++, pPixel++)
                            {
                                iLabel = *pPixel;
                                if (iLabel==iLabelWanted)
                                {
                                    s.sample.x = (int16_t)pt.x;
                                    s.sample.y = (int16_t)pt.y;

                                    samples.push_back(s);
                                }
                            }
                        }
                    }
                    else {
                        vector<cv::Point> vectPoints;
                        unsigned int iSample, iRandom;

                        vectPoints.reserve(vectNbSamplesPerImageLabel[iImg][iLabelWanted]);

                        for (pt.y=0; pt.y<iHeight; pt.y++)
                        {
                            for (pt.x=0, pPixel = labImg.ptr(pt.y); pt.x<iWidth; pt.x++, pPixel++)
                            {
                                iLabel = *pPixel;
                                if (iLabel==iLabelWanted)
                                    vectPoints.push_back(pt);
                            }
                        }

                        if (vectPoints.size()<vectNbSamplesPerImageLabelWanted[iImg][iLabelWanted])
                        {
                            cout<<"WARNING : Image "<<iImg<<" Label "<<(unsigned int)iLabelWanted<<endl;
                            cout<<"  "<<vectNbSamplesPerImageLabelWanted[iImg][iLabelWanted]<<" samples wanted. "<<
                                vectPoints.size()<< " samples found"<<endl;
                        }

                        for (iSample=0; iSample<vectNbSamplesPerImageLabelWanted[iImg][iLabelWanted]; iSample++)
                        {
                            iRandom = rand()%vectPoints.size();

                            s.sample.x = (int16_t)vectPoints[iRandom].x;
                            s.sample.y = (int16_t)vectPoints[iRandom].y;
                            samples.push_back(s);
                        }
                    }
                }
            }
        }

        cout<<"Nb samples ="<<samples.size()<<endl;
    }

  void printDataDistribution(const vector<LabelledSample<Sample<FeatureType>, Label> > &samples) const
  {
    vector<uint32_t> sampDist(nClasses, 0);
    // for(auto it = samples.begin(); it != samples.end(); ++it)
    for(typename vector<LabelledSample<Sample<FeatureType>, Label> >::iterator it = samples.begin(); it != samples.end(); ++it)
      ++sampDist[it->label.value];

    cout << endl << "Total number of " << samples.size() << " are distributed as follows:" << endl;
    for(int i = 0; i < nClasses; ++i)
      cout << "Class " << i << ": " << sampDist[i] << "\tsamples" << endl;

    cout << endl;
  }

  virtual void updateLabelImportance(const vector<LabelledSample<Sample<FeatureType>, Label> > &samples)
  {
    typename vector<LabelledSample<Sample<FeatureType>, Label> >::const_iterator sampleIt = samples.begin();
    typename vector<LabelledSample<Sample<FeatureType>, Label> >::const_iterator sampleEnd = samples.end();

    importance.resize(nClasses);
    fill(importance.begin(), importance.end(), 0);
    for (; sampleIt != sampleEnd; ++sampleIt)
    {
      ++importance[sampleIt->label.value];
    }

    for (uint16_t i = 0; i < nClasses; ++i)
      importance[i] = (importance[i] > 0) ? ((float) samples.size() / importance[i]) : 1.f;
  }

protected:

  uint16_t nClasses;
  uint16_t minSamples;
  uint16_t maxDepth;
  uint32_t nTrials;
  SAMPLING_TYPE samplingType;
  bool interleaved;
  uint32_t samplingParam;
  uint16_t maxProbeOffset;

  const TrainingSet<FeatureType> *ts;
  vector<float> importance;
  mutable RNG rng;
};

}

#endif /* SEMANTICSEGMENTATIONFORESTS_H_ */
