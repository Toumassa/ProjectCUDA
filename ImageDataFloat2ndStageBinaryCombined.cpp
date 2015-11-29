#include <limits>
#include "ImageDataFloat2ndStageBinaryCombined.h"
#include "Global.h"
// #include "graphicwnd.h"

using namespace std;
// using namespace cv;

namespace vision
{

bool ImageDataFloat2ndStageBinaryCombined::setConfiguration(ConfigReader &cfg)
{
    double scaleFactor = cfg.rescaleFactor;
    vector<string>::iterator it, end;
    unsigned int iImg, iFeature;
    cv::Mat imgInput;
    cv::Mat imgSegmap1stStage, imgLabel;
    CImageCacheElement *pImgElem;
    char strPostfix[100];
    string strInputLabelPath;

    // iNbMaxImagesLoaded = 10;
    // bUseIntegralImages = true;

    pData1stStage = new ImageDataFloat();
    pData1stStage->bGenerateFeatures = false;

    pData1stStage->setConfiguration(cfg);

    it = cfg.imageFilenames.begin();
    end = cfg.imageFilenames.end();

    vectImageData.resize(end-it);

    if (bGenerateFeatures==true)
        cout << "Set paths and generate HOT1 features for " << end-it << " images: "<<endl;
    else
        cout << "Just set paths for " << end-it << " images: "<<endl;

    iNbLabels = cfg.numLabels;
    iNbFeatures = pData1stStage->getNbFeatures() + iNbLabels;

    iWidth = pData1stStage->getWidth();
    iHeight = pData1stStage->getHeight();

    // load image data
    for(iImg = 0; it != end; ++it, ++iImg)
    {
        pImgElem = &(vectImageData[iImg]);

        sprintf(strPostfix, "%04d", iImg);

        pImgElem->strInputImage = cfg.imageFolder + "/" + *it;
        pImgElem->strFeatureImagesPath = cfg.feature2ndStageFolder + "/features" + strPostfix;
        pImgElem->strFeatureImagesIntegralPath = cfg.feature2ndStageFolder + "/features_integral" + strPostfix;
        pImgElem->strLabelImagePath = cfg.outputFolder + "/segmap_1st_stage" + strPostfix + ".png";

        if (bGenerateFeatures==true)
        {
            cout<<"Generating 2nd-stage HOT1 features for image "<<iImg<<endl;

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

            computeFeatures(imgSegmap1stStage, pImgElem->vectFeatures);

            if (bUseIntegralImages==true)
            {
                pImgElem->vectFeaturesIntegral.resize(pImgElem->vectFeatures.size());
                for (iFeature=0; iFeature<pImgElem->vectFeatures.size(); iFeature++)
                {
                    cv::integral(pImgElem->vectFeatures[iFeature], pImgElem->vectFeaturesIntegral[iFeature], CV_32F);
                    /*
                    if (pImgElem->vectFeaturesIntegral[iFeature].rows!=pImgElem->vectFeatures[iFeature].rows+1
                        || pImgElem->vectFeaturesIntegral[iFeature].cols!=pImgElem->vectFeatures[iFeature].cols+1)
                    {
                        cout<<"Size differ"<<endl;
                    }*/
                }
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


bool ImageDataFloat2ndStageBinaryCombined::CloseImageData(unsigned int iImg)
{
    CImageCacheElement *pImgElem;

    pImgElem = &(vectImageData[iImg]);

    if (pImgElem->bLoaded==true)
    {
        pImgElem->vectFeatures.clear();
        pImgElem->vectFeaturesIntegral.clear();
        pImgElem->bLoaded = false;
    }

    return true;
}

bool ImageDataFloat2ndStageBinaryCombined::WriteImageData(unsigned int iImg)
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

bool ImageDataFloat2ndStageBinaryCombined::ReadImageData(unsigned int iImg)
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

    if (pImgElem->vectFeatures.size()!=iNbLabels)
        pImgElem->vectFeatures.resize(iNbLabels);

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
        if (pImgElem->vectFeaturesIntegral.size()!=iNbLabels)
            pImgElem->vectFeaturesIntegral.resize(iNbLabels);

        for (iFeature=0; iFeature<pImgElem->vectFeaturesIntegral.size(); iFeature++)
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

cv::Mat *ImageDataFloat2ndStageBinaryCombined::getLabelImage(unsigned int iImg)
{
    return pData1stStage->getLabelImage(iImg);
}

void ImageDataFloat2ndStageBinaryCombined::computeFeatures
    (const cv::Mat &input, vector<cv::Mat> &imgFeatures) const
{
    cv::Point pt;
    int iFeature;

    if (imgFeatures.size()!=iNbLabels)
        imgFeatures.resize(iNbLabels);

    for (iFeature=0; iFeature<iNbLabels; iFeature++)
    {
        if (imgFeatures[iFeature].type()!=CV_32F || imgFeatures[iFeature].cols!=input.cols || imgFeatures[iFeature].rows!=input.rows)
            imgFeatures[iFeature].create(input.rows, input.cols, CV_32F);
    }

    if (input.type()!=CV_8UC1)
    {
        cout<<"ERROR in ImageDataFloat2ndStageBinaryCombined::computeFeatures(...): input image is not 8-bit"<<endl;
        exit(-1);
    }

    for (pt.y=0; pt.y<input.rows; pt.y++)
    {
        for (pt.x=0; pt.x<input.cols; pt.x++)
        {
            for (iFeature=0; iFeature<iNbLabels; iFeature++)
                imgFeatures[iFeature].at<float>(pt) = 0.0f;
            imgFeatures[input.at<unsigned char>(pt)].at<float>(pt) = 1.0f;
        }
    }
}


}


