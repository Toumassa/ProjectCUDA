#include <limits>
#include "ImageDataFloat2ndStageCombined.h"
#include "Global.h"
#include "labelfeature.h"
// #include "graphicwnd.h"

using namespace std;
// using namespace cv;

namespace vision
{

bool ImageDataFloat2ndStageCombined::setConfiguration(ConfigReader &cfg)
{
    double scaleFactor = cfg.rescaleFactor;
    vector<string>::iterator it, end;
    unsigned int iImg, iFeature;
    cv::Mat imgInput;
    cv::Mat imgSegmap1stStage, imgLabel;
    CImageCacheElement *pImgElem;
    char strPostfix[100];
    string strInputLabelPath;
    CLabelFeature labelFeature;

    // iNbMaxImagesLoaded = 10;
    // bUseIntegralImages = true;

    pData1stStage = new ImageDataFloat();
    pData1stStage->bGenerateFeatures = false;

    pData1stStage->setConfiguration(cfg);

    it = cfg.imageFilenames.begin();
    end = cfg.imageFilenames.end();

    vectImageData.resize(end-it);

    if (bGenerateFeatures==true)
        cout << "Set paths and generate label features for " << end-it << " images: "<<endl;
    else
        cout << "Just set paths for " << end-it << " images: "<<endl;

    iNbLabels = cfg.numLabels;
    iNbScales = NO_REGION_SCALES;
    iNbFeatures = pData1stStage->getNbFeatures() + regFeatures;

    iWidth = pData1stStage->getWidth();
    iHeight = pData1stStage->getHeight();

    // load image data
    for(iImg = 0; it != end; ++it, ++iImg)
    {
        pImgElem = &(vectImageData[iImg]);

        sprintf(strPostfix, "%04d", iImg);

        pImgElem->strInputImage = cfg.imageFolder + "/" + *it;
        pImgElem->strFeatureImagesPath = cfg.feature2ndStageFolder + "/features" + strPostfix;
        pImgElem->strFeatureImagesIntegralPath = cfg.feature2ndStageFolder + "/label_integral" + strPostfix;
        pImgElem->strLabelImagePath = cfg.outputFolder + "/segmap_1st_stage" + strPostfix + ".png";

        if (bGenerateFeatures==true)
        {
            cout<<"Generating 2nd-stage features for image "<<iImg<<endl;

            imgSegmap1stStage = cv::imread(pImgElem->strLabelImagePath, cv::IMREAD_GRAYSCALE);
            if (imgSegmap1stStage.data==NULL)
            {
                cout<<"Failed to read 1st-stage segmentation map "<<iImg<<": "<<pImgElem->strLabelImagePath<<endl;
                return false;
            }

            // Check if segmentation map and ground truth image have the same size
            strInputLabelPath = cfg.groundTruthFolder + "/label_rearranged" + strPostfix + ".png";
            imgLabel = cv::imread(strInputLabelPath, cv::IMREAD_GRAYSCALE);
            if (imgLabel.data==NULL)
            {
                cout<<"Failed to read ground truth image "<<iImg<<": "<<pImgElem->strLabelImagePath<<endl;
                return false;
            }

            if (imgSegmap1stStage.size()!=imgLabel.size())
            {
                cv::resize(imgSegmap1stStage, imgSegmap1stStage, imgLabel.size(), 0, 0, cv::INTER_NEAREST);
                cout<<"Segmentation map "<<iImg<<" resized"<<endl;
            }

            labelFeature.SetImage(imgSegmap1stStage);
            pImgElem->vectFeaturesIntegral.resize(6*NO_LABELS);

            unsigned int iLabel;

            for (iLabel=0; iLabel<NO_LABELS; iLabel++)
            {
                pImgElem->vectFeaturesIntegral[iLabel*6 + 0] = labelFeature.arrayIntegralImages[iLabel].one;
                pImgElem->vectFeaturesIntegral[iLabel*6 + 1] = labelFeature.arrayIntegralImages[iLabel].x;
                pImgElem->vectFeaturesIntegral[iLabel*6 + 2] = labelFeature.arrayIntegralImages[iLabel].y;
                pImgElem->vectFeaturesIntegral[iLabel*6 + 3] = labelFeature.arrayIntegralImages[iLabel].xx;
                pImgElem->vectFeaturesIntegral[iLabel*6 + 4] = labelFeature.arrayIntegralImages[iLabel].yy;
                pImgElem->vectFeaturesIntegral[iLabel*6 + 5] = labelFeature.arrayIntegralImages[iLabel].xy;
            }

            pImgElem->bLoaded = true;
            if (WriteImageData(iImg)==false)
                return false;

            CloseImageData(iImg);
        }
        else {
            pImgElem->bLoaded = false;
        }
    }

    cout<<"2nd-stage image data initialized"<<endl;

    return true;
}


bool ImageDataFloat2ndStageCombined::CloseImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==true)
    {
        pImgElem->vectFeaturesIntegral.clear();
        pImgElem->bLoaded = false;
    }

    return true;
}

bool ImageDataFloat2ndStageCombined::WriteImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    unsigned int iLabel, iIntegral;
    char strLabel[10];
    const char strPostfixes[6][10] = {"_one.dat", "_x.dat", "_y.dat", "_xx.dat", "_yy.dat", "_xy.dat"};
    string strFilename;

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==false)
    {
        cout<<"ImageDataFloat2ndStageCombined: cannot write label integral images "<<iImg<<": not loaded"<<endl;
        return false;
    }

    if (pImgElem->vectFeaturesIntegral.size()!=6*NO_LABELS)
    {
        cout<<"ImageDataFloat2ndStageCombined: cannot write label integral images "<<iImg<<": vector does not have the required size"<<endl;
        return false;
    }

    for (iLabel=0; iLabel<NO_LABELS; iLabel++)
    {
        sprintf(strLabel, "_%02d", iLabel);
        for (iIntegral=0; iIntegral<6; iIntegral++)
        {
            strFilename = pImgElem->strFeatureImagesIntegralPath + strLabel + strPostfixes[iIntegral];
            if (WriteImageIntOrFloat(pImgElem->vectFeaturesIntegral[iLabel*6 + iIntegral],
                strFilename.c_str())==false)
            {
                cout<<"ERROR in ImageDataFloat2ndStageCombined::WriteImageData(...):";
                cout<<" failed to write label integral image to "<<strFilename<<endl;
                return false;
            }
        }
    }

    return true;
}

bool ImageDataFloat2ndStageCombined::ReadImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    unsigned int iLabel, iIntegral;
    char strLabel[10];
    const char strPostfixes[6][10] = {"_one.dat", "_x.dat", "_y.dat", "_xx.dat", "_yy.dat", "_xy.dat"};
    string strFilename;

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==true)
    {
        cout<<"ImageDataFloat2ndStageCombined: Do not need to read feature images "<<iImg<<": already loaded"<<endl;
        return false;
    }

    if (pImgElem->vectFeaturesIntegral.size()!=6*NO_LABELS)
        pImgElem->vectFeaturesIntegral.resize(6*NO_LABELS);

    for (iLabel=0; iLabel<NO_LABELS; iLabel++)
    {
        sprintf(strLabel, "_%02d", iLabel);
        for (iIntegral=0; iIntegral<6; iIntegral++)
        {
            strFilename = pImgElem->strFeatureImagesIntegralPath + strLabel + strPostfixes[iIntegral];

            pImgElem->vectFeaturesIntegral[iLabel*6 + iIntegral].create(iHeight, iWidth, CV_32F);
            if (ReadImageIntOrFloat(pImgElem->vectFeaturesIntegral[iLabel*6 + iIntegral],
                strFilename.c_str())==false)
            {
                cout<<"ERROR in ImageDataFloat2ndStageCombined::ReadImageData(...):";
                cout<<" failed to load label integral image "<<strFilename<<endl;
                return false;
            }
        }
    }

    return true;
}

vector<cv::Mat> *ImageDataFloat2ndStageCombined::getFeatureImages(unsigned int iImg)
{
    cout<<"ERROR in ImageDataFloat2ndStageCombined::getFeatureImages(...): function deprecated"<<endl;
    exit(-1);
    return NULL;
}

vector<cv::Mat> *ImageDataFloat2ndStageCombined::getFeatureIntegralImages(unsigned int iImg)
{
    cout<<"ERROR in ImageDataFloat2ndStageCombined::getFeatureIntegralImages(...): function deprecated"<<endl;
    exit(-1);
    return NULL;
}

vector<cv::Mat> *ImageDataFloat2ndStageCombined::getLabelIntegralImages(unsigned int iImg)
{
    CImageCacheElement *pImgElem;
    vector<cv::Mat> *pLabelIntegrals = NULL;

    pImgElem = &(vectImageData[iImg]);
    if (pImgElem->bLoaded==true)
        pLabelIntegrals = &(pImgElem->vectFeaturesIntegral);
    else {
        cout<<"Loading integrals (one,x,y,xx,yy,xy) of image "<<iImg<<endl;
        if (ReadImageData(iImg)==true)
        {
            pImgElem->bLoaded = true;

            pLabelIntegrals = &(pImgElem->vectFeaturesIntegral);

            listIndicesImagesLastLoaded.push_back(iImg);
            while (listIndicesImagesLastLoaded.size()>iNbMaxImagesLoaded)
            {
                CloseImageData(listIndicesImagesLastLoaded.front());
                listIndicesImagesLastLoaded.pop_front();
            }
        }
    }

    return pLabelIntegrals;
}

cv::Mat *ImageDataFloat2ndStageCombined::getLabelImage(unsigned int iImg)
{
    return pData1stStage->getLabelImage(iImg);
}

void ImageDataFloat2ndStageCombined::computeFeatures
    (const cv::Mat &input, vector<cv::Mat> &imgFeatures) const
{
    CLabelFeature labelFeature;
    cv::Point pt;
    float *features = new float[iNbFeatures];
    int iFeature;

    if (imgFeatures.size()!=iNbFeatures)
        imgFeatures.resize(iNbFeatures);

    for (iFeature=0; iFeature<iNbFeatures; iFeature++)
    {
        if (imgFeatures[iFeature].type()!=CV_32F || imgFeatures[iFeature].cols!=input.cols || imgFeatures[iFeature].rows!=input.rows)
            imgFeatures[iFeature].create(input.rows, input.cols, CV_32F);
    }

    if (input.type()!=CV_8UC1)
    {
        cout<<"ERROR in ImageDataFloat2ndStageCombined::computeFeatures(...): input image is not 8-bit"<<endl;
        exit(-1);
    }

    labelFeature.SetImage(input);

    for (pt.y=0; pt.y<input.rows; pt.y++)
    {
        for (pt.x=0; pt.x<input.cols; pt.x++)
        {
            labelFeature.MakeFeatureVector(pt, features);
            for (iFeature=0; iFeature<iNbFeatures; iFeature++)
                imgFeatures[iFeature].at<float>(pt) = features[iFeature];
        }
    }

    delete[] features;
}


}


