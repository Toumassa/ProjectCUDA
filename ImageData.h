// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulo', H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulo'
//
//    Parts of the code (for data representation) in this file use the publicly available code of
//    http://www.vision.ee.ethz.ch/~gallju/projects/houghforest/
//    J. Gall and V. Lempitsky: Class-Speciﬁc Hough Forests for Object Detection. In (CVPR), 2009.
//
//    and the Sigma Points code of Kluckner et al.
//    http://www.icg.tugraz.at/Members/kluckner/files/CovSigmaPointsComp.zip
//    S. Kluckner, T. Mauthner, P.M. Roth and H. Bischof. Semantic Classification in Aerial
//    Imagery by Integrating Appearance and Height Information. In (ACCV), 2009.
//
//
//    October 2013
//
// =========================================================================================

#ifndef IMAGEDATA_H_
#define IMAGEDATA_H_

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ConfigReader.h"
#include "IntegralStructures.h"

#define USE_CORR_COEFF 1

using namespace std;

namespace vision
{
class HoG
{
public:
  HoG();
  ~HoG()
  {
    cvReleaseMat(&Gauss);
    delete ptGauss;
  }
  void extractOBin(cv::Mat Iorient, cv::Mat Imagn, vector<cv::Mat> &out, int off);
private:

  void calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, double* desc);
  void binning(float v, float w, double* desc, int maxb);

  int bins;
  float binsize;

  int g_w;
  CvMat* Gauss;

  // Gauss as vector
  float* ptGauss;
};

static HoG hog;

class ImageData
{
  // Types
  protected:
    typedef struct
    {
        bool bLoaded;
        string strInputImage;
        string strFeatureImagesPath;
        string strFeatureImagesIntegralPath;
        string strLabelImagePath;

        vector<cv::Mat> vectFeatures;
        vector<cv::Mat> vectFeaturesIntegral;
        cv::Mat imgLabel;
    } CImageCacheElement;

  // Member variables
  protected:
    vector<CImageCacheElement> vectImageData;
    string strFormat;

    int iFirstFrame;
    int iNbFeatures;

    int iWidth, iHeight;
    int iNbFps;

    list<unsigned int> listIndicesImagesLastLoaded;
    bool bUseIntegralImages;

  public:
    bool bGenerateFeatures;
    unsigned int iNbMaxImagesLoaded;

  // Member functions
  public:

    ImageData()
    {
        iNbMaxImagesLoaded = 100;
        bUseIntegralImages = true;
        bGenerateFeatures = false;
    }

    virtual bool setConfiguration(ConfigReader &);

    ~ImageData()
    {
    }

    virtual size_t getNbImages() const
    {
        return vectImageData.size();
    }

    virtual int getNbFeatures() const
    {
        return iNbFeatures;
    }

    virtual int getWidth() const
    {
        return iWidth;
    }

    virtual int getHeight() const
    {
        return iHeight;
    }

    virtual vector<cv::Mat> *getFeatureImages(unsigned int);
    virtual vector<cv::Mat> *getFeatureIntegralImages(unsigned int);
    virtual cv::Mat *getLabelImage(unsigned int);
    virtual string getInputImageName(unsigned int);

    virtual void getGradients(const cv::Mat, vector<cv::Mat>&) const;
    virtual cv::Mat matchHistograms(const cv::Mat, const cv::Mat) const;

    virtual bool UseIntegralImages() const {return bUseIntegralImages;}

protected:
    virtual bool CloseImageData(unsigned int);
    virtual bool WriteImageData(unsigned int);
    virtual bool ReadImageData(unsigned int);

    virtual bool CloseAllImageData();

    static inline bool fileExists(const string& filename)
    {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
      return true;
    }
    return false;
    }

    virtual void computeFeatures(const cv::Mat &, vector<cv::Mat> &imgFeatures) const;
    virtual void computeFeaturesWithCorrCoeff(const cv::Mat &input, vector<cv::Mat> &imgFeatures) const;
    virtual void computeHOGLike4SingleChannel(const cv::Mat &img, vector<cv::Mat> &vImg, int offset,
      bool include_first_order_deriv, bool include_second_order_deriv) const;

    virtual bool WriteImageIntOrFloat(const cv::Mat &, const char *) const;
    virtual bool ReadImageIntOrFloat(cv::Mat &, const char *) const;

    // vector<vector<cv::Mat> > imageData;
    // vector<cv::Mat> gtImages;
};

class ImageDataSort
{
  // Types
  public:
    class VideoStat
    {
      public:
        string strVideoName;
        vector<unsigned int> vectImagesIndices;
        vector<unsigned int> vectNbSamplesPerLabel;

      public:
        VideoStat() {}
    };

  // Member variables
  public:
    ImageData *pImageData;
    vector<VideoStat> vectVideoStats;
    unsigned int iNbLabels;

  public:
    ImageDataSort(unsigned int);
    bool SetData(ImageData *);
    void GenerateRandomSequence(vector<unsigned int> &);
    void RandomSplit_TrainValidationTest();

    void GenerateRandomSequence_TrainingImagesIndices(vector<vector<unsigned int> > &);
};

}
#endif
