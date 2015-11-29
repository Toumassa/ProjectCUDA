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

#ifndef IMAGEDATAFLOAT_H_
#define IMAGEDATAFLOAT_H_

#include "ImageData.h"

namespace vision
{
class HoGFloat
{
public:
  HoGFloat();
  ~HoGFloat()
  {
    cvReleaseMat(&Gauss);
    delete ptGauss;
  }
  void extractOBin(const cv::Mat &Iorient, const cv::Mat &Imagn, vector<cv::Mat>& out, int off);
private:

  void calcHoGBin(const float* ptOrient, const float* ptMagn, int width, float* desc);
  void binning(const float v, const float w, float* desc, int maxb);

  int bins;
  float binsize;

  int g_w;
  CvMat* Gauss;

  // Gauss as vector
  float* ptGauss;
};

static HoGFloat hogFloat;

class ImageDataFloat : public ImageData
{
  protected:
    virtual bool WriteImageData(unsigned int);
    virtual bool ReadImageData(unsigned int);

  public:
    ImageDataFloat():ImageData() {}

    virtual bool setConfiguration(ConfigReader &);

    virtual void getGradients(const cv::Mat, vector<cv::Mat> &) const;
    virtual cv::Mat matchHistograms(const cv::Mat, const cv::Mat) const;

    virtual void computeFeatures(const cv::Mat &, vector<cv::Mat>& imgFeatures) const;

    virtual void computeFeaturesWithCorrCoeff(const cv::Mat &, vector<cv::Mat>& imgFeatures) const;

    virtual void computeHOGLike4SingleChannel(const cv::Mat &img, vector<cv::Mat>& vImg, int offset,
        bool include_first_order_deriv, bool include_second_order_deriv) const;
};

}

#endif
