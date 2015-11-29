#include <limits>
#include "ImageDataFloat.h"
#include "Global.h"
// #include "graphicwnd.h"

using namespace std;
using namespace cv;

namespace vision
{
  bool ImageDataFloat::setConfiguration(ConfigReader &cfg)
  {
    double scaleFactor = cfg.rescaleFactor;
    vector<string>::iterator it, end;
    unsigned int iImg, iFeature;
    cv::Mat imgInput, imgLabel;
    CImageCacheElement *pImgElem;
    char strPostfix[100];

    it = cfg.imageFilenames.begin();
    end = cfg.imageFilenames.end();

    if (bGenerateFeatures==true)
        cout << "Set paths and generate HoG features for " << end-it << " images: "<<endl;
    else
        cout << "Just set paths for " << end-it << " images: "<<endl;

    vectImageData.resize(end-it);

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
                    cv::integral(pImgElem->vectFeatures[iFeature], pImgElem->vectFeaturesIntegral[iFeature], CV_32F);
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


bool ImageDataFloat::WriteImageData(unsigned int iImg)
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
        sprintf(strPostfix, "_%03d.dat", iFeature);
        if (WriteImageIntOrFloat(pImgElem->vectFeatures[iFeature], (pImgElem->strFeatureImagesPath + strPostfix).c_str())==false)
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

    return true;
}

bool ImageDataFloat::ReadImageData(unsigned int iImg)
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
        pImgElem->vectFeatures[iFeature].create(iHeight, iWidth, CV_32F);
        sprintf(strPostfix, "_%03d.dat", iFeature);
        if (ReadImageIntOrFloat(pImgElem->vectFeatures[iFeature], (pImgElem->strFeatureImagesPath + strPostfix).c_str())==false)
        {
            cout<<"Cannot read feature image ("<<iImg<<","<<iFeature<<") to "<<pImgElem->strFeatureImagesPath + strPostfix<<endl;
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
                pImgElem->vectFeatures[iFeature].rows+1, pImgElem->vectFeatures[iFeature].cols+1, CV_32F);

            sprintf(strPostfix, "_%03d.dat", iFeature);
            if (ReadImageIntOrFloat(pImgElem->vectFeaturesIntegral[iFeature], (pImgElem->strFeatureImagesIntegralPath + strPostfix).c_str())==false)
            {
                cout<<"Cannot read feature image integral ("<<iImg<<","<<iFeature<<") to "<<pImgElem->strFeatureImagesIntegralPath + strPostfix<<endl;
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

    return true;
}

void ImageDataFloat::computeFeatures(const cv::Mat &input, vector<cv::Mat> &imgFeatures) const
{
    cv::Mat inputF;

    input.convertTo(inputF, CV_32FC3);

    // get LAB channels
    cv::GaussianBlur(inputF, inputF, Size(5,5), 1.0);
    cv::cvtColor(inputF, inputF, COLOR_LBGR2Lab);
    cv::split(inputF, imgFeatures);

    // L, a, b, L_x, L_y, L_xx, L_yy, HOG_L
    imgFeatures.resize(3 + 4 + 9);
    for(unsigned int c = 3; c < imgFeatures.size(); ++c)
        imgFeatures[c] = cv::Mat::zeros(input.size(), CV_32F);

    int offset = 3;
    computeHOGLike4SingleChannel(imgFeatures[0], imgFeatures, offset, true, true);
}


void ImageDataFloat::computeFeaturesWithCorrCoeff(const cv::Mat &input, vector<cv::Mat>& imgFeatures) const
{
    cv::Mat inputF;

    input.convertTo(inputF, CV_32FC3);
    inputF /= 255.0f;

    // get LAB channels
    cv::Mat bgrClone, labImg;

    cv::GaussianBlur(inputF, bgrClone, cv::Size(5,5), 1.0);
    cv::cvtColor(bgrClone, labImg, COLOR_LBGR2Lab);
    cv::split(labImg, imgFeatures);

    // CArray2DFloatWnd::Show(imgFeatures[0], "L");
    // CArray2DFloatWnd::Show(imgFeatures[1], "a");
    // CArray2DFloatWnd::Show(imgFeatures[2], "b");
    // equalize L channel
    // equalizeHist(imgFeatures[0], imgFeatures[0]);

    int numChannels = 5; // number of channels used for correlation coefficient computation

    // L, L_x, L_y, L_xx, L_yy, HOG_L
    imgFeatures.resize(1 + 4 + 9 + (int)((numChannels * (numChannels-1)) / 2));
    for(unsigned int c = 3; c < imgFeatures.size(); ++c)
      imgFeatures[c] = cv::Mat::zeros(input.size(), CV_32F);

    int offset = 1;
    computeHOGLike4SingleChannel(imgFeatures[0], imgFeatures, offset, true, true);

    // compute correlation coefficients between B,G,R,L_dx,L_dy
    offset = 14;
    vector<cv::Mat> tmpSplit;
    split(bgrClone, tmpSplit);            // B G R

    // equalize each of the RGB channels
    //for(int i = 0; i < tmpSplit.size(); ++i)
    //  equalizeHist(tmpSplit[i], tmpSplit[i]);

    tmpSplit.resize(numChannels);
    tmpSplit[3] = imgFeatures[1].clone(); // L_x
    tmpSplit[4] = imgFeatures[2].clone(); // L_y

    // copy data into float array
    float *cov_feat_input = new float[input.rows * input.cols * numChannels]; //B G R I_x I_y -> results in 10 covariance feature channels
    float *d_ptr = cov_feat_input;
    for(int c = 0; c < tmpSplit.size(); ++c)
      for(int h = 0; h < input.rows; ++h)
        for(int w = 0; w < input.cols; ++w, ++d_ptr)
          *d_ptr = (float)tmpSplit[c].at<float>(h, w);

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

            imgFeatures[idx].at<float>(h,w) = (1.0f+corr_coef)*127.5f;
          }
      }

      delete [] cov;
      delete [] mean;
      delete integralStructure;
}



void ImageDataFloat::computeHOGLike4SingleChannel(const cv::Mat &img, vector<cv::Mat> & vImg, int offset,
                                            bool include_first_order_deriv, bool include_second_order_deriv) const
{
    // Temporary images for computing I_x, I_y (Avoid overflow for cvSobel)
    cv::Mat I_x = cv::Mat(img.size(), CV_32F);
    cv::Mat I_y = cv::Mat(img.size(), CV_32F);

    cv::Mat Itmp1 = cv::Mat::zeros(img.size(), CV_32F); // orientation info
    cv::Mat Itmp2 = img.clone(); // magnitude info

    // |I_x|, |I_y|
    cv::Sobel(img, I_x, CV_32F, 1, 0);
    cv::Sobel(img, I_y, CV_32F, 0, 1);

    int l_offset = offset;

    if(include_first_order_deriv)
    {
        // I_x.convertTo(vImg[l_offset], CV_32F, 0.25);
        // I_y.convertTo(vImg[l_offset+1], CV_32F, 0.25);
        vImg[l_offset] = I_x.clone();
        vImg[l_offset+1] = I_y.clone();

        l_offset += 2;
    }


    // compute orientation information
    float *dataX, *dataY, *pAngle, *pMag;

    for(size_t y = 0; y < Itmp1.rows; ++y)
    {
        dataX = (float *)I_x.ptr(y);
        dataY = (float *)I_y.ptr(y);
        pAngle = (float *)Itmp1.ptr(y);
        pMag = (float *)Itmp2.ptr(y);

    //  dataX += I_x.step1(0), dataY += I_y.step1(0), dataZ += Itmp1.step1(0))
        for(size_t x = 0; x < Itmp1.cols; ++x)
        {
            // Avoid division by zero
            float tx = *dataX + (float)_copysign(0.000001f, *dataX);

            // Scaling [-pi/2 pi/2] -> [0 80*pi]
            *pAngle = ( atanf(*dataY/tx)+ CV_PI / 2.0f ) * 80.0f;

            // *pAngle = atan2f(*dataY, *dataX); ?
            *pMag = sqrtf(*dataX*(*dataX) + *dataY*(*dataY));

            dataX++;
            dataY++;
            pAngle++;
            pMag++;
        }
    }


  // 9-bin HOG feature stored at vImg - offset
  if(include_second_order_deriv)
    hogFloat.extractOBin(Itmp1, Itmp2, vImg, l_offset+2);
  else
    hogFloat.extractOBin(Itmp1, Itmp2, vImg, l_offset);

  if(include_second_order_deriv)
  {
    // |I_xx|, |I_yy|
    /*
    Sobel(img,I_x,CV_16S,2,0);
    I_x.convertTo(vImg[l_offset], CV_8UC1, 0.25);

    Sobel(img,I_y,CV_16S,0,2);
    I_y.convertTo(vImg[l_offset+1], CV_8UC1, 0.25);
    */
    cv::Sobel(img, vImg[l_offset], CV_32F, 2, 0);
    cv::Sobel(img, vImg[l_offset+1], CV_32F, 0, 2);
  }
}


/////////////////////////////////////////////////////////////////////////////
// histogram matching
cv::Mat ImageDataFloat::matchHistograms(const cv::Mat img, const cv::Mat destHgram) const
{
  // destHgrm needs to be 1 x m

  const int m = 256;
  // get (cumulative) histograms
  Mat hist = Mat::zeros(Size(m,1), CV_32FC1);
  Mat cumHist = hist.clone();

  Mat locDestHGram = destHgram * ((float)img.cols*img.rows)/(sum(destHgram)[0]);
  Mat cumDest = Mat::zeros(destHgram.size(), CV_32FC1);

  // setup histogram of input image
  float *histPtr = hist.ptr<float>();
  MatConstIterator_<unsigned char> it = img.begin<unsigned char>();
  MatConstIterator_<unsigned char> end = img.end<unsigned char>();
  while(it != end)
  {
    ++histPtr[*it];
    ++it;
  }

  // get cumsum of input image and destination histogram simultaneously
  MatIterator_<float> cIt = cumHist.begin<float>();
  MatConstIterator_<float> destIt = locDestHGram.begin<float>();
  MatIterator_<float> cDestIt = cumDest.begin<float>();
  MatIterator_<float> cEnd = cumHist.end<float>();
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
  Mat tmp = hist.clone();
  tmp.at<float>(0,0) = 0;
  tmp.at<float>(0,m-1) = 0;

  Mat tol = Mat::ones(m, 1, CV_32FC1) * (tmp / 2);
  Mat err = (cumHist.t() * Mat::ones(1,m,CV_32FC1) - Mat::ones(m,1,CV_32FC1) * cumDest) + tol;

  float *errPtr = err.ptr<float>();

  MatIterator_<float> errIt = err.begin<float>();
  MatIterator_<float> errEnd = err.end<float>();
  float subst = (float)img.cols * img.rows;
  float minVal = numeric_limits<float>::epsilon() * subst;
  while(errIt != errEnd)
  {
    if(*errIt < -minVal)
      *errIt = subst;
    ++errIt;
  }

  Mat T = Mat::zeros(m,1,CV_32FC1);
  int minIdx;
  for(int i = 0; i < m; ++i)
  {
    minIdx = 0;
    for(int j = 1; j < m; ++j)
      minIdx = (err.at<float>(i,j) < err.at<float>(i,minIdx)) ? j : minIdx;

    T.at<float>(i,0) = (float)minIdx;
  }

  Mat res = Mat::zeros(img.size(), CV_8UC1);
  for(int y = 0; y < img.rows; ++y)
    for(int x = 0; x < img.cols; ++x)
      res.at<unsigned char>(y,x) = saturate_cast<unsigned char>(T.at<float>(img.at<unsigned char>(y,x),0));

  return res;
}



//////////////////////////////////////////////////////
//                   get gradients                  //
//////////////////////////////////////////////////////

void ImageDataFloat::getGradients(const Mat input, vector<Mat>& grad) const
{
  enum neighborhood {N, NE, E, SE, S, SW, W, NW};

  Mat in, tmp;
  input.convertTo(tmp, CV_32FC1);
  GaussianBlur(tmp, in, Size(3,3), 3, 3);

  grad.clear();
  for(size_t i = 0; i < 8; ++i)
    grad.push_back(Mat::zeros(input.size(), CV_32FC1));

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



HoGFloat::HoGFloat()
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

void HoGFloat::extractOBin(const cv::Mat &Iorient, const cv::Mat &Imagn, vector<cv::Mat>& out, int off)
{
    float* desc = new float[bins];
    float** ptOut     = new float*[bins];
    int l; // Bin index

    /*
    float** ptOut_row = new float*[bins];

    for(int k=off; k<bins+off; ++k) {
        out[k].setTo(0); //reset output image
        ptOut[k-off] = (float*)out[k].data; // get pointer to data
    }

    // get pointers to orientation, magnitude
    float* ptOrient; // = (uchar*)Iorient.data;
    float* ptMagn; // = (uchar*)Imagn.data;
    float* ptOrient_row, *ptMagn_row;

    int step = (int)Iorient.step1(0);

    int off_w = int(g_w/2.0);
    for(int l=0; l<bins; ++l)
        ptOut[l] += off_w*step;

    for(int y=0;y<Iorient.rows-g_w; y++) // , ptMagn+=step, ptOrient+=step)
    {
        // Get row pointers
        ptOrient_row = (float *)Iorient.ptr(y); // &ptOrient[0];
        ptMagn_row = (float *)Imagn.ptr(y); // &ptMagn[0];

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
    */

    const float* ptOrient; // = (uchar*)Iorient.data;
    const float* ptMagn; // = (uchar*)Imagn.data;

    for (int y=g_w/2;y<Iorient.rows-g_w/2; y++) // , ptMagn+=step, ptOrient+=step)
    {
        // Get row pointers
        ptOrient = (const float *)Iorient.ptr(y) + g_w/2; // &ptOrient[0];
        ptMagn = (const float *)Imagn.ptr(y) + g_w/2; // &ptMagn[0];

        for (l=0; l<bins; ++l)
            ptOut[l] = (float *)out[off+l].ptr(y) + g_w/2;

        for (int x=g_w/2; x<Iorient.cols-g_w/2; ++x)
        {
            calcHoGBin( ptOrient, ptMagn, Iorient.cols, desc );
            for(l=0; l<bins; ++l)
            {
                *(ptOut[l]) = desc[l];
                ++ptOut[l];
            }
            ++ptOrient;
            ++ptMagn;
        }
    }

    delete[] desc;
    delete[] ptOut;
    // delete[] ptOut_row;
}

inline void HoGFloat::calcHoGBin(const float* ptOrient, const float* ptMagn, int width, float* desc) {
	for(int i=0; i<bins;i++)
		desc[i]=0.0f;

	const float* ptO = ptOrient - g_w/2*width - g_w/2;
	const float* ptM = ptMagn - g_w/2*width - g_w/2;

	int i=0;
	for(int y=0;y<g_w; ++y, ptO+=width, ptM+=width) {
		for(int x=0;x<g_w; ++x, ++i) {
			binning(ptO[x]/binsize, ptM[x] * ptGauss[i], desc, bins);
		}
	}
}

inline void HoGFloat::binning(const float v, const float w, float* desc, int maxb) {
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


}
