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
//    J. Gall and V. Lempitsky: Class-Speci?c Hough Forests for Object Detection. In (CVPR), 2009.
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

#include "ImageData.h"
#include "Global.h"
#include <limits>
// #include "graphicwnd.h"

using namespace std;

namespace vision
{

bool ImageData::setConfiguration(ConfigReader &cfg)
{
    double scaleFactor = cfg.rescaleFactor;
    vector<string>::iterator it, end;
    unsigned int iImg, iFeature;
    cv::Mat imgInput, imgLabel;
    CImageCacheElement *pImgElem;
    char strPostfix[100];

    it = cfg.imageFilenames.begin();
    end = cfg.imageFilenames.end();

    vectImageData.resize(end-it);

    if (bGenerateFeatures==true)
        cout << "Set paths and generate HoG features for " << end-it << " images: "<<endl;
    else
        cout << "Just set paths for " << end-it << " images: "<<endl;

    #if USE_CORR_COEFF
    iNbFeatures = 24;
    #else
    iNbFeatures = 16;
    #endif

    // load image data
    for (iImg = 0; it != end; ++it, ++iImg)
    {
        pImgElem = &(vectImageData[iImg]);

        sprintf(strPostfix, "%04d", iImg);

        pImgElem->strInputImage = cfg.imageFolder + "/" + *it;
        pImgElem->strFeatureImagesPath = cfg.featureFolder + "/features" + strPostfix;
        pImgElem->strFeatureImagesIntegralPath = cfg.featureFolder + "/features_integral" + strPostfix;
        pImgElem->strLabelImagePath = cfg.groundTruthFolder + "/label_rearranged" + strPostfix + ".png";

        imgLabel = cv::imread(pImgElem->strLabelImagePath, cv::IMREAD_GRAYSCALE);
        if (imgLabel.data==NULL)
        {
            cout<<"Failed to read ground truth image "<<iImg<<": "<<pImgElem->strLabelImagePath<<endl;
            return false;
        }

        // This is the first image we read, we can set the size of image data
        if (iImg==0)
        {
            iWidth = imgLabel.cols;
            iHeight = imgLabel.rows;
        }

        if (bGenerateFeatures==true)
        {
            cout<<"Generating features for image "<<iImg<<endl;

            imgInput = cv::imread(pImgElem->strInputImage, cv::IMREAD_COLOR);
            if (imgInput.data==NULL)
            {
                cout<<"Failed to read input image "<<iImg<<": "<<pImgElem->strInputImage<<endl;
                return false;
            }

            cv::resize(imgInput, imgInput, cv::Size(), scaleFactor, scaleFactor);

            if (imgInput.cols!=iWidth || imgInput.rows!=iHeight)
            {
                cout<<"Scaled input image and ground truth image have different sizes. Did you set the correct scale factor?"<<endl;
                return false;
            }

            #if USE_CORR_COEFF
            computeFeaturesWithCorrCoeff(imgInput, pImgElem->vectFeatures);
            #else
            computeFeatures(imgInput, pImgElem->vectFeatures);
            #endif

            if (bUseIntegralImages==true)
            {
                pImgElem->vectFeaturesIntegral.resize(pImgElem->vectFeatures.size());
                for (iFeature=0; iFeature<pImgElem->vectFeatures.size(); iFeature++)
                    cv::integral(pImgElem->vectFeatures[iFeature], pImgElem->vectFeaturesIntegral[iFeature], CV_32S);
            }

            pImgElem->bLoaded = true;

            WriteImageData(iImg);
            CloseImageData(iImg);
        }
        else
            pImgElem->bLoaded = false;
    }

    cout<<"Image data initialized"<<endl;

    return true;
}

vector<cv::Mat> *ImageData::getFeatureImages(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    vector<cv::Mat> *pFeatures = NULL;

    pImgElem = &(vectImageData[iImg]);
    if (pImgElem->bLoaded==true)
    {
        //if (bUseIntegralImages==true)
        //    pFeatures = &(pImgElem->vectFeaturesIntegral);
        // else
        pFeatures = &(pImgElem->vectFeatures);
    }
    else {
        cout<<"Loading features of image "<<iImg<<endl;
        if (ReadImageData(iImg)==true)
        {
            pImgElem->bLoaded = true;

            pFeatures = &(pImgElem->vectFeatures);

            listIndicesImagesLastLoaded.push_back(iImg);
            while (listIndicesImagesLastLoaded.size()>iNbMaxImagesLoaded)
            {
                // cout<<"ImageData::getFeatureImages(int): closing image "<<listIndicesImagesLastLoaded.front()<<endl;
                CloseImageData(listIndicesImagesLastLoaded.front());
                listIndicesImagesLastLoaded.pop_front();
            }
        }
    }

    return pFeatures;
}

vector<cv::Mat> *ImageData::getFeatureIntegralImages(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    vector<cv::Mat> *pFeatures = NULL;

    assert(bUseIntegralImages==true);

    pImgElem = &(vectImageData[iImg]);
    if (pImgElem->bLoaded==true)
    {
        pFeatures = &(pImgElem->vectFeaturesIntegral);
    }
    else {
        cout<<"Loading features of image "<<iImg<<endl;
        if (ReadImageData(iImg)==true)
        {
            pImgElem->bLoaded = true;

            pFeatures = &(pImgElem->vectFeaturesIntegral);

            listIndicesImagesLastLoaded.push_back(iImg);
            while (listIndicesImagesLastLoaded.size()>iNbMaxImagesLoaded)
            {
                // cout<<"ImageData::getFeatureImages(int): closing image "<<listIndicesImagesLastLoaded.front()<<endl;
                CloseImageData(listIndicesImagesLastLoaded.front());
                listIndicesImagesLastLoaded.pop_front();
            }
        }
    }

    return pFeatures;
}

cv::Mat *ImageData::getLabelImage(unsigned int iImg)
{
    CImageCacheElement *pImgElem;

    pImgElem = &(vectImageData[iImg]);
    if (pImgElem->bLoaded==true)
        return &(pImgElem->imgLabel);
    else {
        cout<<"Loading features of image "<<iImg<<endl;
        if (ReadImageData(iImg)==true)
        {
            pImgElem->bLoaded = true;

            listIndicesImagesLastLoaded.push_back(iImg);
            while (listIndicesImagesLastLoaded.size()>iNbMaxImagesLoaded)
            {
                // cout<<"ImageData::getLabelImage(int): closing image "<<listIndicesImagesLastLoaded.front()<<endl;
                CloseImageData(listIndicesImagesLastLoaded.front());
                listIndicesImagesLastLoaded.pop_front();
            }
            return &(pImgElem->imgLabel);
        }
        else {
            return NULL;
        }
    }
}

string ImageData::getInputImageName(unsigned int iImg)
{
    return vectImageData[iImg].strInputImage;
}

bool ImageData::CloseImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==true)
    {
        pImgElem->vectFeatures.clear();
        if (bUseIntegralImages==true)
            pImgElem->vectFeaturesIntegral.clear();
        pImgElem->imgLabel.release();
        pImgElem->bLoaded = false;
    }

    return true;
}

bool ImageData::WriteImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    unsigned int iFeature;
    char strPostfix[100];

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==false)
    {
        cout<<"Cannot write feature images "<<iImg<<": not loaded"<<endl;
        return false;
    }
    for (iFeature=0; iFeature<pImgElem->vectFeatures.size(); iFeature++)
    {
        sprintf(strPostfix, "_%03d.png", iFeature);
        if (cv::imwrite(pImgElem->strFeatureImagesPath + strPostfix, pImgElem->vectFeatures[iFeature])==false)
        {
            cout<<"Cannot write feature image ("<<iImg<<","<<iFeature<<") to "<<pImgElem->strFeatureImagesPath + strPostfix<<endl;
            return false;
        }
    }

    if (bUseIntegralImages==true)
    {
        for (iFeature=0; iFeature<pImgElem->vectFeaturesIntegral.size(); iFeature++)
        {
            sprintf(strPostfix, "_%03d.dat", iFeature);
            if (WriteImageIntOrFloat(pImgElem->vectFeaturesIntegral[iFeature], (pImgElem->strFeatureImagesIntegralPath + strPostfix).c_str())==false)
            {
                cout<<"Cannot write feature image integral ("<<iImg<<","<<iFeature<<") to "<<pImgElem->strFeatureImagesIntegralPath + strPostfix<<endl;
                return false;
            }
        }
    }

    if (cv::imwrite(pImgElem->strLabelImagePath, pImgElem->imgLabel)==false)
    {
        cout<<"Cannot write label image "<<iImg<<" to "<<pImgElem->strLabelImagePath<<endl;
        return false;
    }

    return true;
}

bool ImageData::ReadImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    unsigned int iFeature;
    char strPostfix[100];

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==true)
    {
        cout<<"Do not need to read feature images "<<iImg<<": already loaded"<<endl;
        return false;
    }

    if (pImgElem->vectFeatures.size()!=iNbFeatures)
        pImgElem->vectFeatures.resize(iNbFeatures);

    for (iFeature=0; iFeature<pImgElem->vectFeatures.size(); iFeature++)
    {
        sprintf(strPostfix, "_%03d.png", iFeature);

        pImgElem->vectFeatures[iFeature] = cv::imread(pImgElem->strFeatureImagesPath + strPostfix, cv::IMREAD_GRAYSCALE);
        if (pImgElem->vectFeatures[iFeature].data==NULL)
        {
            pImgElem->vectFeatures.clear();
            cout<<"Cannot read feature image ("<<iImg<<","<<iFeature<<") from "<<pImgElem->strFeatureImagesPath + strPostfix<<endl;
            return false;
        }
    }

    if (bUseIntegralImages==true)
    {
        if (pImgElem->vectFeaturesIntegral.size()!=iNbFeatures)
            pImgElem->vectFeaturesIntegral.resize(iNbFeatures);

        for (iFeature=0; iFeature<pImgElem->vectFeatures.size(); iFeature++)
        {
            pImgElem->vectFeaturesIntegral[iFeature].create(
                pImgElem->vectFeatures[iFeature].rows+1, pImgElem->vectFeatures[iFeature].cols+1, CV_32S);

            sprintf(strPostfix, "_%03d.dat", iFeature);
            if (ReadImageIntOrFloat(pImgElem->vectFeaturesIntegral[iFeature], (pImgElem->strFeatureImagesIntegralPath + strPostfix).c_str())==false)
            {
                cout<<"Cannot read feature image integral ("<<iImg<<","<<iFeature<<") from "<<pImgElem->strFeatureImagesIntegralPath + strPostfix<<endl;
                return false;
            }
        }
    }

    pImgElem->imgLabel = cv::imread(pImgElem->strLabelImagePath, cv::IMREAD_GRAYSCALE);
    if (pImgElem->imgLabel.data==NULL)
    {
        pImgElem->vectFeatures.clear();
        cout<<"Cannot read label image "<<iImg<<pImgElem->strLabelImagePath<<endl;
        return false;
    }
    /*
    unsigned char *pLabel;
    cv::Point pt;
    for (pt.y=0; pt.y<pImgElem->imgLabel.rows; pt.y++)
    {
        pLabel = pImgElem->imgLabel.ptr(pt.y);
        for (pt.x=0; pt.x<pImgElem->imgLabel.cols; pt.x++)
        {
            if (*pLabel==0 || *pLabel==3 || *pLabel==4 || *pLabel==7 || *pLabel==13)
                *pLabel = 11;
            pLabel++;
        }
    }*/

    return true;
}

bool ImageData::CloseAllImageData()
{
    CImageCacheElement *pImgElem;
    unsigned int iImg;

    for (iImg=0; iImg<vectImageData.size(); iImg++)
    {
        pImgElem = &(vectImageData[iImg]);

        if (pImgElem->bLoaded==true)
        {
            pImgElem->vectFeatures.clear();
            if (bUseIntegralImages==true)
                pImgElem->vectFeaturesIntegral.clear();
            pImgElem->imgLabel.release();
            pImgElem->bLoaded = false;
        }
    }
    listIndicesImagesLastLoaded.clear();

    return true;
}


bool ImageData::WriteImageIntOrFloat(const cv::Mat &imgInt, const char *strFilename) const
{
    FILE *pFile;

    if (imgInt.type()!=CV_32S && imgInt.type()!=CV_32F)
    {
        cout<<"ERROR in ImageData::WriteImageIntoOrFloat(...): incompatible type"<<endl;
        return false;
    }

    pFile = fopen(strFilename, "wb");
    if (pFile==NULL)
        return false;

    if (fwrite(imgInt.data, 4, imgInt.rows*imgInt.cols, pFile)<imgInt.rows*imgInt.cols)
    {
        fclose(pFile);
        return false;
    }

    fclose(pFile);
    return true;
}

bool ImageData::ReadImageIntOrFloat(cv::Mat &imgInt, const char *strFilename) const
{
    FILE *pFile;

    if (imgInt.type()!=CV_32S && imgInt.type()!=CV_32F)
    {
        cout<<"ERROR in ImageData::WriteImageIntOrFloat(...): incompatible type"<<endl;
        return false;
    }

    pFile = fopen(strFilename, "rb");
    if (pFile==NULL)
        return false;

    if (fread(imgInt.data, 4, imgInt.rows*imgInt.cols, pFile)<imgInt.rows*imgInt.cols)
    {
        fclose(pFile);
        return false;
    }

    fclose(pFile);
    return true;
}

void ImageData::computeFeatures(const cv::Mat &input, vector<cv::Mat>& imgFeatures) const
{
    // get LAB channels
    cv::GaussianBlur(input, input, cv::Size(5,5), 1.0);
    cv::cvtColor(input, input, cv::COLOR_LBGR2Lab);
    cv::split(input, imgFeatures);

    // L, a, b, L_x, L_y, L_xx, L_yy, HOG_L
    imgFeatures.resize(3 + 4 + 9);
    for(unsigned int c = 3; c < imgFeatures.size(); ++c)
      imgFeatures[c] = cv::Mat::zeros(input.size(), CV_8UC1);

    int offset = 3;
    computeHOGLike4SingleChannel(imgFeatures[0], imgFeatures, offset, true, true);

#if 0
    static int save_it = 1;
    if(save_it)
    {
      // for debugging only
      char buffer[40];
      for(unsigned int i = 0; i<imgFeatures.size();++i) {
        sprintf(buffer,"c:/temp/aaa/out-%02d.png",i);
        imwrite( string(buffer), imgFeatures[i] );
      }

      save_it = 0;
    }
#endif
  }


void ImageData::computeFeaturesWithCorrCoeff(const cv::Mat &input, vector<cv::Mat> &imgFeatures) const
{
    // get LAB channels
    cv::Mat bgrClone, labImg;
    cv::GaussianBlur(input, bgrClone, cv::Size(5,5), 1.0);
    cv::cvtColor(bgrClone, labImg, cv::COLOR_LBGR2Lab);
    cv::split(labImg, imgFeatures);

    // equalize L channel
    cv::equalizeHist(imgFeatures[0], imgFeatures[0]);

    int numChannels = 5; // number of channels used for correlation coefficient computation

    // L, L_x, L_y, L_xx, L_yy, HOG_L
    imgFeatures.resize(1 + 4 + 9 + (int)((numChannels * (numChannels-1)) / 2));
    for(unsigned int c = 3; c < imgFeatures.size(); ++c)
      imgFeatures[c] = cv::Mat::zeros(input.size(), CV_8UC1);

    int offset = 1;
    computeHOGLike4SingleChannel(imgFeatures[0], imgFeatures, offset, true, true);

    // compute correlation coefficients between B,G,R,L_dx,L_dy
    offset = 14;
    vector<cv::Mat> tmpSplit;
    cv::split(bgrClone, tmpSplit);            // B G R

    // equalize each of the RGB channels
    for(int i = 0; i < tmpSplit.size(); ++i)
        cv::equalizeHist(tmpSplit[i], tmpSplit[i]);

    tmpSplit.resize(numChannels);
    tmpSplit[3] = imgFeatures[1].clone(); // L_x
    tmpSplit[4] = imgFeatures[2].clone(); // L_y

    // copy data into float array
    float *cov_feat_input = new float[input.rows * input.cols * numChannels]; //B G R I_x I_y -> results in 10 covariance feature channels
    float *d_ptr = cov_feat_input;
    for(int c = 0; c < tmpSplit.size(); ++c)
      for(int h = 0; h < input.rows; ++h)
        for(int w = 0; w < input.cols; ++w, ++d_ptr)
          *d_ptr = (float)tmpSplit[c].at<unsigned char>(h, w);

    IntegralStructures<float, float>* integralStructure = new IntegralStructures<float, float>(cov_feat_input, numChannels, input.rows, input.cols, 1|2);
    delete [] cov_feat_input;

    int w_off = 10;
    int h_off = 10;

    int num_cov = numChannels * numChannels;
    float *cov = new float[num_cov];
    float *mean = new float[numChannels];

    for(int w = w_off; w < input.cols-w_off; ++w)
      for(int h = h_off; h < input.rows-h_off; ++h)
      {
        integralStructure->GetMeanAndVariance(myRect(h-h_off, w-w_off, 2*h_off, 2*w_off), mean, cov);

        // set standard deviation on main diagonal
        for(int c = 0; c < numChannels; ++c)
          cov[c*(numChannels+1)] = (cov[c*(numChannels+1)] > 0.) ? sqrt(cov[c*(numChannels+1)]) : 0;

        // compute correlation coefficients
        int idx = offset;
        float corr_coef;
        for(int c = 0; c < numChannels; ++c)
          for(int r = c+1; r < numChannels; ++r, ++idx)
          {
            if(cov[c*(numChannels+1)] == 0. || cov[r*(numChannels+1)] == 0.)
            {
              corr_coef = 0.;
            }
            else
            {
              corr_coef = cov[c*numChannels + r] / (cov[c*(numChannels+1)]*cov[r*(numChannels+1)]);
              // check bounds
              corr_coef = MIN(1, MAX(-1, corr_coef));
            }

            // re-normalize to [0..255]

            imgFeatures[idx].at<unsigned char>(h,w) = cv::saturate_cast<unsigned char>((1.+corr_coef)*127.5);
          }
      }

      delete [] cov;
      delete [] mean;
      delete integralStructure;

#if 0
    static int save_it = 1;
    if(save_it)
    {
      // for debugging only
      char buffer[40];
      for(unsigned int i = 0; i<imgFeatures.size();++i) {
        sprintf(buffer,"c:/temp/ccc/out-%02d.png",i);
        imwrite( string(buffer), imgFeatures[i] );
      }

      save_it = 0;
    }
#endif
  }



void ImageData::computeHOGLike4SingleChannel(const cv::Mat &img, vector<cv::Mat>& vImg, int offset,
                                             bool include_first_order_deriv, bool include_second_order_deriv) const
{
  // Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
  cv::Mat I_x = cv::Mat(img.size(), CV_16SC1);
  cv::Mat I_y = cv::Mat(img.size(), CV_16SC1);

  cv::Mat Itmp1 = cv::Mat::zeros(img.size(), CV_8UC1); // orientation info
  cv::Mat Itmp2 = img.clone(); // magnitude info

  // |I_x|, |I_y|
  cv::Sobel(img, I_x, CV_16S, 1, 0);
  cv::Sobel(img, I_y, CV_16S, 0, 1);

  int l_offset = offset;

  if(include_first_order_deriv)
  {
    I_x.convertTo(vImg[l_offset], CV_8UC1, 0.25);
    I_y.convertTo(vImg[l_offset+1], CV_8UC1, 0.25);
    l_offset += 2;
  }

  {
    // compute orientation information
    short* dataX = (short*)I_x.data;
    short* dataY = (short*)I_y.data;
    uchar* dataZ = (uchar*)Itmp1.data;
    for(size_t y = 0; y < Itmp1.rows; ++y, dataX += I_x.step1(0), dataY += I_y.step1(0), dataZ += Itmp1.step1(0))
      for(size_t x = 0; x < Itmp1.cols; ++x)
      {
        // Avoid division by zero
        float tx = (float)dataX[x] + (float)_copysign(0.000001f, (float)dataX[x]);
        // Scaling [-pi/2 pi/2] -> [0 80*pi]
        dataZ[x]=uchar( ( atan((float)dataY[x]/tx)+ CV_PI / 2.0f ) * 80 );
      }
  }


  {
    // compute magnitude information
    short* dataX = (short*)I_x.data;
    short* dataY = (short*)I_y.data;
    uchar* dataZ = (uchar*)Itmp2.data;
    for(size_t y = 0; y < Itmp2.rows; ++y, dataX += I_x.step1(0), dataY += I_y.step1(0), dataZ += Itmp1.step1(0))
      for(size_t x = 0; x < Itmp2.cols; ++x)
        dataZ[x] = (uchar)( sqrt((float)dataX[x]*(float)dataX[x] + (float)dataY[x]*(float)dataY[x]) );
  }

  // 9-bin HOG feature stored at vImg - offset
  if(include_second_order_deriv)
    hog.extractOBin(Itmp1, Itmp2, vImg, l_offset+2);
  else
    hog.extractOBin(Itmp1, Itmp2, vImg, l_offset);

  if(include_second_order_deriv)
  {
    // |I_xx|, |I_yy|
    cv::Sobel(img,I_x,CV_16S,2,0);
    I_x.convertTo(vImg[l_offset], CV_8UC1, 0.25);

    cv::Sobel(img,I_y,CV_16S,0,2);
    I_y.convertTo(vImg[l_offset+1], CV_8UC1, 0.25);
  }
}


/////////////////////////////////////////////////////////////////////////////
// histogram matching
cv::Mat ImageData::matchHistograms(const cv::Mat img, const cv::Mat destHgram) const
{
  // destHgrm needs to be 1 x m

  const int m = 256;
  // get (cumulative) histograms
  cv::Mat hist = cv::Mat::zeros(cv::Size(m,1), CV_32FC1);
  cv::Mat cumHist = hist.clone();

  cv::Mat locDestHGram = destHgram * ((float)img.cols*img.rows)/(sum(destHgram)[0]);
  cv::Mat cumDest = cv::Mat::zeros(destHgram.size(), CV_32FC1);

  // setup histogram of input image
  float *histPtr = hist.ptr<float>();
  cv::MatConstIterator_<unsigned char> it = img.begin<unsigned char>();
  cv::MatConstIterator_<unsigned char> end = img.end<unsigned char>();
  while(it != end)
  {
    ++histPtr[*it];
    ++it;
  }

  // get cumsum of input image and destination histogram simultaneously
  cv::MatIterator_<float> cIt = cumHist.begin<float>();
  cv::MatConstIterator_<float> destIt = locDestHGram.begin<float>();
  cv::MatIterator_<float> cDestIt = cumDest.begin<float>();
  cv::MatIterator_<float> cEnd = cumHist.end<float>();
  *cIt = histPtr[0];
  ++cIt;
  ++histPtr;

  *cDestIt = *destIt;
  ++cDestIt;
  ++destIt;

  while(cIt != cEnd)
  {
    *cIt = *(cIt-1) + *histPtr;
    ++cIt; ++histPtr;

    *cDestIt = *(cDestIt-1) + *destIt;
    ++cDestIt; ++destIt;
  }

  cout << "cumhist: " << cumHist << endl << endl;
  cout << "cumDesthist: " << cumDest << endl << endl;


  // get transformation function
  cv::Mat tmp = hist.clone();
  tmp.at<float>(0,0) = 0;
  tmp.at<float>(0,m-1) = 0;

  cv::Mat tol = cv::Mat::ones(m, 1, CV_32FC1) * (tmp / 2);
  cv::Mat err = (cumHist.t() * cv::Mat::ones(1,m,CV_32FC1) - cv::Mat::ones(m,1,CV_32FC1) * cumDest) + tol;

  float *errPtr = err.ptr<float>();

  cv::MatIterator_<float> errIt = err.begin<float>();
  cv::MatIterator_<float> errEnd = err.end<float>();
  float subst = (float)img.cols * img.rows;
  float minVal = numeric_limits<float>::epsilon() * subst;
  while(errIt != errEnd)
  {
    if(*errIt < -minVal)
      *errIt = subst;
    ++errIt;
  }

  cv::Mat T = cv::Mat::zeros(m,1,CV_32FC1);
  int minIdx;
  for(int i = 0; i < m; ++i)
  {
    minIdx = 0;
    for(int j = 1; j < m; ++j)
      minIdx = (err.at<float>(i,j) < err.at<float>(i,minIdx)) ? j : minIdx;

    T.at<float>(i,0) = (float)minIdx;
  }

  cv::Mat res = cv::Mat::zeros(img.size(), CV_8UC1);
  for(int y = 0; y < img.rows; ++y)
    for(int x = 0; x < img.cols; ++x)
      res.at<unsigned char>(y,x) = cv::saturate_cast<unsigned char>(T.at<float>(img.at<unsigned char>(y,x),0));

  return res;
}



//////////////////////////////////////////////////////
//                   get gradients                  //
//////////////////////////////////////////////////////

void ImageData::getGradients(const cv::Mat input, vector<cv::Mat>& grad) const
{
  enum neighborhood {N, NE, E, SE, S, SW, W, NW};

  cv::Mat in, tmp;
  input.convertTo(tmp, CV_32FC1);
  cv::GaussianBlur(tmp, in, cv::Size(3,3), 3, 3);

  grad.clear();
  for(size_t i = 0; i < 8; ++i)
    grad.push_back(cv::Mat::zeros(input.size(), CV_32FC1));

  // handling within borders
  float o;
  for(int y = 1; y < in.rows - 1; ++y)
    for(int x = 1; x < in.cols - 1; ++x)
    {
      o = in.at<float>(y,x);
      grad[N ].at<float>(y,x) = in.at<float>(y-1,x  ) - o;
      grad[NE].at<float>(y,x) = in.at<float>(y-1,x+1) - o;
      grad[E ].at<float>(y,x) = in.at<float>(y  ,x+1) - o;
      grad[SE].at<float>(y,x) = in.at<float>(y+1,x+1) - o;
      grad[S ].at<float>(y,x) = in.at<float>(y+1,x  ) - o;
      grad[SW].at<float>(y,x) = in.at<float>(y+1,x-1) - o;
      grad[W ].at<float>(y,x) = in.at<float>(y  ,x-1) - o;
      grad[NW].at<float>(y,x) = in.at<float>(y-1,x-1) - o;
    }

  // handle vertical borders (left/right)
  int xE = in.cols-1, xEm1 = in.cols-2;
  for(int y = 0; y < in.rows; ++y)
  {
    // left border
    grad[N ].at<float>(y,0) = grad[N ].at<float>(y,1);
    grad[NE].at<float>(y,0) = grad[NE].at<float>(y,1);
    grad[E ].at<float>(y,0) = grad[E ].at<float>(y,1);
    grad[SE].at<float>(y,0) = grad[SE].at<float>(y,1);
    grad[S ].at<float>(y,0) = grad[S ].at<float>(y,1);
    grad[SW].at<float>(y,0) = grad[SW].at<float>(y,1);
    grad[W ].at<float>(y,0) = grad[W ].at<float>(y,1);
    grad[NW].at<float>(y,0) = grad[NW].at<float>(y,1);

    // right border
    grad[N ].at<float>(y,xE) = grad[N ].at<float>(y,xEm1);
    grad[NE].at<float>(y,xE) = grad[NE].at<float>(y,xEm1);
    grad[E ].at<float>(y,xE) = grad[E ].at<float>(y,xEm1);
    grad[SE].at<float>(y,xE) = grad[SE].at<float>(y,xEm1);
    grad[S ].at<float>(y,xE) = grad[S ].at<float>(y,xEm1);
    grad[SW].at<float>(y,xE) = grad[SW].at<float>(y,xEm1);
    grad[W ].at<float>(y,xE) = grad[W ].at<float>(y,xEm1);
    grad[NW].at<float>(y,xE) = grad[NW].at<float>(y,xEm1);
  }

  // handle horizontal borders (top/bottom)
  int yE = in.rows-1, yEm1 = in.rows-2;
  for(int x = 0; x < in.cols; ++x)
  {
    // top border
    grad[N ].at<float>(0,x) = grad[N ].at<float>(1,x);
    grad[NE].at<float>(0,x) = grad[NE].at<float>(1,x);
    grad[E ].at<float>(0,x) = grad[E ].at<float>(1,x);
    grad[SE].at<float>(0,x) = grad[SE].at<float>(1,x);
    grad[S ].at<float>(0,x) = grad[S ].at<float>(1,x);
    grad[SW].at<float>(0,x) = grad[SW].at<float>(1,x);
    grad[W ].at<float>(0,x) = grad[W ].at<float>(1,x);
    grad[NW].at<float>(0,x) = grad[NW].at<float>(1,x);

    // bottom border
    grad[N ].at<float>(yE,x) = grad[N ].at<float>(yEm1,x);
    grad[NE].at<float>(yE,x) = grad[NE].at<float>(yEm1,x);
    grad[E ].at<float>(yE,x) = grad[E ].at<float>(yEm1,x);
    grad[SE].at<float>(yE,x) = grad[SE].at<float>(yEm1,x);
    grad[S ].at<float>(yE,x) = grad[S ].at<float>(yEm1,x);
    grad[SW].at<float>(yE,x) = grad[SW].at<float>(yEm1,x);
    grad[W ].at<float>(yE,x) = grad[W ].at<float>(yEm1,x);
    grad[NW].at<float>(yE,x) = grad[NW].at<float>(yEm1,x);
  }
}

HoG::HoG()
{
	bins = 9;
	binsize = (3.14159265f*80.0f)/float(bins);

	g_w = 5;
	Gauss = cvCreateMat( g_w, g_w, CV_32FC1 );
	double a = -(g_w-1)/2.0;
	double sigma2 = 2*(0.5*g_w)*(0.5*g_w);
	double count = 0;
	for(int x = 0; x<g_w; ++x) {
		for(int y = 0; y<g_w; ++y) {
			double tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
			count += tmp;
			cvSet2D( Gauss, x, y, cvScalar(tmp) );
		}
	}
	cvConvertScale( Gauss, Gauss, 1.0/count);

	ptGauss = new float[g_w*g_w];
	int i = 0;
	for(int y = 0; y<g_w; ++y)
		for(int x = 0; x<g_w; ++x)
			ptGauss[i++] = (float)cvmGet( Gauss, x, y );

}

void HoG::extractOBin(cv::Mat Iorient, cv::Mat Imagn, vector<cv::Mat>& out, int off)
{
	double* desc = new double[bins];

	uchar** ptOut     = new uchar*[bins];
	uchar** ptOut_row = new uchar*[bins];
	for(int k=off; k<bins+off; ++k) {
		out[k].setTo(0); //reset output image
    ptOut[k-off] = (uchar*)out[k].data; // get pointer to data
	}

	// get pointers to orientation, magnitude
	uchar* ptOrient = (uchar*)Iorient.data;
	uchar* ptMagn = (uchar*)Imagn.data;
	uchar* ptOrient_row, *ptMagn_row;
  int step = (int)Iorient.step1(0);

	int off_w = int(g_w/2.0);
	for(int l=0; l<bins; ++l)
		ptOut[l] += off_w*step;

	for(int y=0;y<Iorient.rows-g_w; y++, ptMagn+=step, ptOrient+=step)
  {
		// Get row pointers
		ptOrient_row = &ptOrient[0];
		ptMagn_row = &ptMagn[0];

    for(int l=0; l<bins; ++l)
			ptOut_row[l] = &ptOut[l][0]+off_w;

		for(int x=0; x<Iorient.cols-g_w; ++x, ++ptOrient_row, ++ptMagn_row)
    {
			calcHoGBin( ptOrient_row, ptMagn_row, step, desc );

			for(int l=0; l<bins; ++l) {
				*ptOut_row[l] = (uchar)desc[l];
				++ptOut_row[l];
			}
		}

		// update pointer
		for(int l=0; l<bins; ++l)
			ptOut[l] += step;
	}

	delete[] desc;
	delete[] ptOut;
	delete[] ptOut_row;
}

inline void HoG::calcHoGBin(uchar* ptOrient, uchar* ptMagn, int step, double* desc) {
	for(int i=0; i<bins;i++)
		desc[i]=0;

	uchar* ptO = &ptOrient[0];
	uchar* ptM = &ptMagn[0];
	int i=0;
	for(int y=0;y<g_w; ++y, ptO+=step, ptM+=step) {
		for(int x=0;x<g_w; ++x, ++i) {
			binning((float)ptO[x]/binsize, (float)ptM[x] * ptGauss[i], desc, bins);
		}
	}
}

inline void HoG::binning(float v, float w, double* desc, int maxb) {
	int bin1 = int(v);
	int bin2;
	float delta = v-bin1-0.5f;
	if(delta<0) {
		bin2 = bin1 < 1 ? maxb-1 : bin1-1;
		delta = -delta;
	} else
		bin2 = bin1 < maxb-1 ? bin1+1 : 0;
	desc[bin1] += (1-delta)*w;
	desc[bin2] += delta*w;
}


ImageDataSort::ImageDataSort(unsigned int nblab)
{
    iNbLabels = nblab;
}

bool ImageDataSort::SetData(ImageData *pData)
{
    unsigned int iImg, iVid, iLabel;
    unsigned char *pPixel;
    cv::Mat *pImgLabel;
    cv::Point pt;
    VideoStat *pVS;
    string strInputName, strVideoName;
    string strVideoExt = ".avi";
    size_t pos;

    vectVideoStats.clear();
    pImageData = pData;

    for (iImg=0; iImg<pImageData->getNbImages(); iImg++)
    {
        strInputName = pImageData->getInputImageName(iImg);

        pos = strInputName.find(strVideoExt);

        assert(pos != string::npos);
        strVideoName = strInputName.substr(0, pos); //-strVideoExt.length()+1);

        iVid = 0;
        while (iVid<vectVideoStats.size() && strVideoName!=vectVideoStats[iVid].strVideoName)
            iVid++;
        if (iVid<vectVideoStats.size())
            vectVideoStats[iVid].vectImagesIndices.push_back(iImg);
        else {
            cout<<"New video = "<<strVideoName<<endl;
            vectVideoStats.push_back(VideoStat());
            vectVideoStats.back().strVideoName = strVideoName;
            vectVideoStats.back().vectImagesIndices.push_back(iImg);
        }
    }
    /*
    for (iVid=0; iVid<vectVideoStats.size(); iVid++)
    {
        pVS = &(vectVideoStats[iVid]);
        pVS->vectNbSamplesPerLabel.resize(iNbLabels);
        for (iLabel=0; iLabel<iNbLabels; iLabel++)
            pVS->vectNbSamplesPerLabel[iLabel] = 0;

        for (iImg=0; iImg<pVS->vectImagesIndices.size(); iImg++)
        {
            pImgLabel = pImageData->getLabelImage(pVS->vectImagesIndices[iImg]);

            for (pt.y=0; pt.y<pImgLabel->rows; pt.y++)
            {
                pPixel = pImgLabel->ptr(pt.y);
                for (pt.x=0; pt.x<pImgLabel->cols; pt.x++, pPixel++)
                {
                    iLabel = (unsigned int)(*pPixel);
                    //if (iLabel==0 || iLabel==3 || iLabel==4 || iLabel==7 || iLabel==13)
                    //    pVS->vectNbSamplesPerLabel[11]++;
                    //else
                    pVS->vectNbSamplesPerLabel[iLabel]++;
                }
            }
        }

        cout<<iVid<<": "<<pVS->strVideoName<<endl;
        //for (iLabel=0; iLabel<iNbLabels; iLabel++)
        //    cout<<iLabel<<": "<< pVS->vectNbSamplesPerLabel[iLabel]<<" samples"<<endl;
    }*/
}

void ImageDataSort::GenerateRandomSequence(vector<unsigned int> &vectIndices)
{
    unsigned int iNbVideos = vectVideoStats.size();
    unsigned int iIndex, iIndex2, iTemp;
    unsigned int iPermutation, iNbPermutations = 20;

    vectIndices.clear();

    for (iIndex=0; iIndex<iNbVideos; iIndex++)
        vectIndices.push_back(iIndex);

    for (iPermutation=0; iPermutation<iNbPermutations; iPermutation++)
    {
        iIndex = rand()%iNbVideos;
        iIndex2 = rand()%iNbVideos;

        iTemp = vectIndices[iIndex];
        vectIndices[iIndex] = vectIndices[iIndex2];
        vectIndices[iIndex2] = iTemp;
    }
}

void ImageDataSort::RandomSplit_TrainValidationTest()
{
    unsigned int iTry, iNbTrials = 50;
    vector<unsigned int> seqCurrent, seqBest;
    float costCurrent, costBest;

    const VideoStat *pVS;
    vector<float> distribTraining, distribValidation, distribTest;

    unsigned int iLabel;
    unsigned int iVid, iNbVideos = vectVideoStats.size();
    unsigned int iNbSamplesTraining, iNbSamplesValidation, iNbSamplesTest;
    unsigned int iNbVideosTraining, iNbVideosValidation, iNbVideosTest;

    iNbVideosTraining = iNbVideos*3/5;
    iNbVideosValidation = iNbVideos/5;
    iNbVideosTest = iNbVideos - iNbVideosTraining - iNbVideosValidation;

    distribTraining.resize(iNbLabels);
    distribValidation.resize(iNbLabels);
    distribTest.resize(iNbLabels);

    costBest = FLT_MAX;
    for (iTry=0; iTry<iNbTrials; iTry++)
    {
        GenerateRandomSequence(seqCurrent);

        iNbSamplesTraining = 0;
        iNbSamplesValidation = 0;
        iNbSamplesTest = 0;

        for (iLabel=0; iLabel<iNbLabels; iLabel++)
        {
            distribTraining[iLabel] = 0.0f;
            distribValidation[iLabel] = 0.0f;
            distribTest[iLabel] = 0.0f;
        }

        // Training
        for (iVid = 0; iVid<iNbVideosTraining; iVid++)
        {
            pVS = &(vectVideoStats[seqCurrent[iVid]]);
            iNbSamplesTraining += pVS->vectImagesIndices.size()*640*480;

            for (iLabel=0; iLabel<iNbLabels; iLabel++)
                distribTraining[iLabel] += (float)pVS->vectNbSamplesPerLabel[iLabel];
        }

        // Validation
        for (iVid = 0; iVid<iNbVideosValidation; iVid++)
        {
            pVS = &(vectVideoStats[seqCurrent[iNbVideosTraining+iVid]]);
            iNbSamplesValidation += pVS->vectImagesIndices.size()*640*480;

            for (iLabel=0; iLabel<iNbLabels; iLabel++)
                distribValidation[iLabel] += (float)pVS->vectNbSamplesPerLabel[iLabel];
        }

        // Test
        for (iVid = 0; iVid<iNbVideosTest; iVid++)
        {
            pVS = &(vectVideoStats[seqCurrent[iNbVideosTraining+iNbVideosValidation+iVid]]);
            iNbSamplesTest += pVS->vectImagesIndices.size()*640*480;

            for (iLabel=0; iLabel<iNbLabels; iLabel++)
                distribTest[iLabel] += (float)pVS->vectNbSamplesPerLabel[iLabel];
        }

        // Make probability distributions
        costCurrent = 0.0f;
        for (iLabel=0; iLabel<iNbLabels; iLabel++)
        {
            distribTraining[iLabel] /= iNbSamplesTraining;
            distribValidation[iLabel] /= iNbSamplesValidation;
            distribTest[iLabel] /= iNbSamplesTest;

            costCurrent += fabs(distribTraining[iLabel] - distribValidation[iLabel]);
            costCurrent += fabs(distribValidation[iLabel] - distribTest[iLabel]);
        }

        if (costCurrent<costBest)
        {
            seqBest = seqCurrent;
            costBest = costCurrent;

            cout<<"Best combination (cost = "<<costBest<<")"<<endl;
            for (iVid=0; iVid<seqBest.size(); iVid++)
                cout<<seqBest[iVid]<<" ";
            cout<<endl;

            for (iLabel=0; iLabel<iNbLabels; iLabel++)
                cout<<iLabel<<"  "<<distribTraining[iLabel]<<"  "<<distribValidation[iLabel]
                    <<"  "<<distribTest[iLabel]<<endl;
        }
    }

    ofstream of("./repartition_training_validation_test.txt");

    of<<"TRAINING"<<endl;
    for (iVid = 0; iVid<iNbVideosTraining; iVid++)
    {
        pVS = &(vectVideoStats[seqBest[iVid]]);
        of<<seqBest[iVid]<<": "<<pVS->strVideoName<<endl;
    }

    of<<endl<<"VALIDATION"<<endl;
    for (iVid = 0; iVid<iNbVideosValidation; iVid++)
    {
        pVS = &(vectVideoStats[seqBest[iNbVideosTraining+iVid]]);
        of<<seqBest[iNbVideosTraining+iVid]<<": "<<pVS->strVideoName<<endl;
    }

    of<<endl<<"TEST"<<endl;
    for (iVid = 0; iVid<iNbVideosTest; iVid++)
    {
        pVS = &(vectVideoStats[seqBest[iNbVideosTraining+iNbVideosValidation+iVid]]);
        of<<seqBest[iNbVideosTraining+iNbVideosValidation+iVid]<<": "<<pVS->strVideoName<<endl;
    }

    of.close();
}

void ImageDataSort::GenerateRandomSequence_TrainingImagesIndices(vector<vector<unsigned int> > &vectTrainingImagesIndicesPerTree)
{
    unsigned int arrayIndicesTrainingVideos[12] = {0, 2, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18};
    unsigned int iVid, iImg, iNbTrainingVideos = 12;
    unsigned int iNbTrainingImages = 0;
    unsigned int iTree, iNbTrees = 6;
    const VideoStat *pVS;

    for (iVid=0; iVid<iNbTrainingVideos; iVid++)
    {
        pVS = &(vectVideoStats[arrayIndicesTrainingVideos[iVid]]);
        iNbTrainingImages += pVS->vectImagesIndices.size();
    }

    vectTrainingImagesIndicesPerTree.resize(iNbTrees);
    // vectTrainingImagesIndices.reserve(iNbTrainingImages);

    for (iVid=0; iVid<iNbTrainingVideos; iVid++)
    {
        pVS = &(vectVideoStats[arrayIndicesTrainingVideos[iVid]]);
        // cout<<pVS->strVideoName<<endl;
        for (iImg=0; iImg<pVS->vectImagesIndices.size(); iImg++)
        {
            iTree = rand()%iNbTrees;
            vectTrainingImagesIndicesPerTree[iTree].push_back(pVS->vectImagesIndices[iImg]);
        }
        // cout<<endl;
    }

    for (iTree=0; iTree<iNbTrees; iTree++)
    {
        char strFilename[100];

        sprintf(strFilename, "./training_idx_tree%d.txt", iTree+1);
        FILE *pFile = fopen(strFilename, "w");

        cout<<"Tree "<<iTree<<"  "<<vectTrainingImagesIndicesPerTree[iTree].size()<<" images "<<endl;
        for (iImg=0; iImg<vectTrainingImagesIndicesPerTree[iTree].size(); iImg++)
            fprintf(pFile, "%d\n", vectTrainingImagesIndicesPerTree[iTree][iImg]);
        cout<<vectTrainingImagesIndicesPerTree[iTree][iImg]<<" ";
        cout<<endl;
        fclose(pFile);
    }
}

}
