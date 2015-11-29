#ifndef IMAGEDATA2NDSTAGECOMBINED_H_
#define IMAGEDATA2NDSTAGECOMBINED_H_

#include "ImageDataFloat.h"

using namespace std;

namespace vision
{

class ImageDataFloat2ndStageCombined : public ImageDataFloat
{
  // Member variables
  protected:
    ImageDataFloat *pData1stStage;
    int iNbScales;
    int iNbLabels;

    // Contains features and interal features (1st stage and 2nd stage) of the last loaded image
    vector<cv::Mat> vectFeatureImagesCombined;
    vector<cv::Mat> vectFeatureIntegralImagesCombined;

  // Member functions
  public:

    ImageDataFloat2ndStageCombined():ImageDataFloat()
    {
        // iNbMaxImagesLoaded = 100;
        // bUseIntegralImages = false;
        pData1stStage = NULL;
    }

    // ImageDataFloat2ndStageCombined(ConfigReader *cfg);
    virtual bool setConfiguration(ConfigReader &);

    ~ImageDataFloat2ndStageCombined()
    {
        if (pData1stStage!=NULL)
            delete pData1stStage;
    }

    virtual ImageDataFloat *GetData1stStage() {return pData1stStage;}

    virtual vector<cv::Mat> *getFeatureImages(unsigned int);
    virtual vector<cv::Mat> *getFeatureIntegralImages(unsigned int);
    virtual vector<cv::Mat> *getLabelIntegralImages(unsigned int);

    virtual cv::Mat *getLabelImage(unsigned int);

protected:
    virtual bool CloseImageData(unsigned int);
    virtual bool WriteImageData(unsigned int);
    virtual bool ReadImageData(unsigned int);

    virtual void computeFeatures(const cv::Mat &, vector<cv::Mat> &imgFeatures) const;
};

}
#endif
