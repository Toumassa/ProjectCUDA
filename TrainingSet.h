#ifndef TRAININGSET_H
#define TRAININGSET_H

#include <iostream>
#include "ImageData.h"
#include "ImageDataFloat.h"
#include "ImageDataFloat2ndStageCombined.h"
#include "ImageDataFloat2ndStageBinaryCombined.h"
#include "labelfeature.h"

using namespace std;

namespace vision {

template <typename FeatureType> class TrainingSet
{
  public:
    TrainingSet(int nblab, ImageData *pData)
    {
        cv::Mat *pLabel;
        // vector<cv::Mat> *pVectFeatures;

        importance.resize(nblab, 1.0);
        // images.reserve(nImages);
        // labellings.reserve(nImages);
        nChannels = 0;
        pImageData = pData;
        nLabels = nblab;

        assert(pImageData->getNbImages()>0);

        pLabel = pImageData->getLabelImage(0);
        assert(pLabel!=NULL);

        iWidth = pLabel->cols;
        iHeight = pLabel->rows;

        // pVectFeatures = pImageData->getFeatureImages(0);
        // assert(pVectFeatures!=NULL);
        nChannels = pImageData->getNbFeatures(); // pVectFeatures->size();
    }

    bool usesIntImgRepresentation() const
    {
        return pImageData->UseIntegralImages();
    }

    virtual FeatureType getValue(uint16_t imageId, uint8_t channel, int16_t x, int16_t y) const
    {
        vector<cv::Mat> *pFeatureImages;

        pFeatureImages = pImageData->getFeatureImages((unsigned int)imageId);
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channel].at<FeatureType>(y, x);
    }

    // returns value from integral image representation (handling of +1 for pt2 coords must be done externally!)
    virtual FeatureType getValueIntegral(uint16_t imageId, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2) const
    {
        if (pImageData->UseIntegralImages()==true)
        {


            vector<cv::Mat> *pFeatureImages;
            cv::Mat *pFeaturesIntegral;

            pFeatureImages = pImageData->getFeatureIntegralImages((unsigned int)imageId);
            assert(pFeatureImages!=NULL);

            pFeaturesIntegral = &((*pFeatureImages)[channel]);
            FeatureType res = (FeatureType)pFeaturesIntegral->at<int32_t>(y2, x2) -
                pFeaturesIntegral->at<int32_t>(y2, x1) -
                pFeaturesIntegral->at<int32_t>(y1, x2) +
                pFeaturesIntegral->at<int32_t>(y1, x1);
        	return res;

        }
        else {

        	std::cerr << "CW: shouldn't we use integral images here???\n";
        	exit (1);

            cv::Point pt;

            vector<cv::Mat> *pFeatureImages;
            cv::Mat *pFeaturesImage;
            uint8_t *pData;

            pFeatureImages = pImageData->getFeatureImages((unsigned int)imageId);
            assert(pFeatureImages!=NULL);

            pFeaturesImage = &((*pFeatureImages)[channel]);

            int32_t sum = 0;
            for (pt.y=y1+1; pt.y<=y2; pt.y++)
            {
                pData = pFeaturesImage->ptr(pt.y);
                for (pt.x=x1+1; pt.x<=x2; pt.x++, pData++)
                    sum += (int32_t)(*pData);
            }
            return sum;
        }
    }

    virtual uint8_t getLabel(uint16_t imageId, int16_t x, int16_t y) const
    {
        cv::Mat *pLabelImage;

        pLabelImage = pImageData->getLabelImage((unsigned int)imageId);
        assert(pLabelImage!=NULL);

        return pLabelImage->at<uint8_t>(y, x);
    }

    virtual cv::Mat getLabelImage(uint16_t imageId) const
    {
        cv::Mat *pLabelImage;

        pLabelImage = pImageData->getLabelImage((unsigned int)imageId);
        assert(pLabelImage!=NULL);

        return *pLabelImage;
    }

    virtual cv::Mat getImageData(uint16_t imageId, uint16_t channelId) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getImageData(uint16_t, uint16_t)"<<endl;
        #endif

	/*
        cout<<imageId<<" ";

        if (imageId!=iLastImgAccess)
        {
            cout<<" CHANGE ";
            iLastImgAccess = imageId;
        }*/

        vector<cv::Mat> *pFeatureImages;

        pFeatureImages = pImageData->getFeatureImages((unsigned int)imageId);
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channelId];
    }

    virtual uint8_t getNLabels() const
    {
        return nLabels;
    }

    virtual uint8_t getNChannels() const
    {
        return nChannels;
    }

    virtual uint16_t getNbImages() const
    {
        return (uint16_t)pImageData->getNbImages();
    }

    virtual uint16_t getImgWidth(uint16_t imageId) const
    {
        /*
        if (pImageData->UseIntegralImages()==true)
            return iWidth - 1;
        else*/
        return iWidth;
    }

    virtual uint16_t getImgHeight(uint16_t imageId) const
    {
        /*
        if (pImageData->UseIntegralImages()==true)
            return iHeight - 1;
        else*/
        return iHeight;
    }

    virtual void setLabelImportance(vector<float> imp)
    {
        importance = imp;
    }

    virtual const vector<float> getLabelImportance() const
    {
        return importance;
    }
    
    ImageData *pImageData;
  protected:
    /*
    vector<vector<cv::Mat> > images;
    vector<cv::Mat> labellings;
    */
    vector<float> importance;
    uint8_t nLabels;
    uint8_t nChannels;
    int iWidth, iHeight;

    // bool integralImgs;
    // FILE *pFileLog;
};

// typedef TTrainingSet<uint8_t> TrainingSet;

template <class FeatureType> class TrainingSetSelection : public TrainingSet<FeatureType>
{
  public:
    TrainingSetSelection(int nblab, ImageData *pData):TrainingSet<FeatureType>(nblab, pData)
    {
    }


    

    virtual FeatureType getValue(uint16_t imageId, uint8_t channel, int16_t x, int16_t y) const
    {
        vector<cv::Mat> *pFeatureImages;

        pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channel].at<FeatureType>(y, x);
    }

    // returns value from integral image representation (handling of +1 for pt2 coords must be done externally!)
    virtual FeatureType getValueIntegral(uint16_t imageId, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2) const
    {
        if (this->pImageData->UseIntegralImages()==true)
        {
            vector<cv::Mat> *pFeatureImages;
            cv::Mat *pFeaturesIntegral;

            pFeatureImages = this->pImageData->getFeatureIntegralImages(vectSelectedImagesIndices[imageId]);
            assert(pFeatureImages!=NULL);

            pFeaturesIntegral = &((*pFeatureImages)[channel]);
            FeatureType res = pFeaturesIntegral->at<FeatureType>(y2, x2) -
                pFeaturesIntegral->at<FeatureType>(y2, x1) -
                pFeaturesIntegral->at<FeatureType>(y1, x2) +
                pFeaturesIntegral->at<FeatureType>(y1, x1);

#ifdef VERBOSE_PREDICTION
        	std::cerr << " [Integral(c=" << (int) channel << ",x1=" << x1 << ",y1=" << y1 
        			  << ",x2=" << x2 << ",y2=" << y2 
        			  << ",w=" << pFeaturesIntegral->cols << ",h=" << pFeaturesIntegral->rows 
        			  << "->" << res << "]";
#endif 
				return res;
        }
        else {
            cv::Point pt;

            vector<cv::Mat> *pFeatureImages;
            cv::Mat *pFeaturesImage;
            FeatureType *pData;

            pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
            assert(pFeatureImages!=NULL);

            pFeaturesImage = &((*pFeatureImages)[channel]);

            FeatureType sum = 0;
            for (pt.y=y1+1; pt.y<=y2; pt.y++)
            {
                pData = (FeatureType *)pFeaturesImage->ptr(pt.y);
                for (pt.x=x1+1; pt.x<=x2; pt.x++, pData++)
                    sum += *pData; //pFeaturesIntegral->at<int32_t>(pt);
            }
            return sum;
        }
    }

    virtual uint8_t getLabel(uint16_t imageId, int16_t x, int16_t y) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabel(uint16_t, int16_t x 2)"<<endl;
        #endif
        cv::Mat *pLabelImage;

        pLabelImage = this->pImageData->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return pLabelImage->at<uint8_t>(y, x);
    }

    virtual cv::Mat getLabelImage(uint16_t imageId) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabelImage(uint16_t)"<<endl;
        #endif

        cv::Mat *pLabelImage;
        // std::cerr << "selected=" << vectSelectedImagesIndices[imageId] << "\n";
        pLabelImage = this->pImageData->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return *pLabelImage;
    }

    virtual cv::Mat getImageData(uint16_t imageId, uint16_t channelId) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getImageData(uint16_t, uint16_t)"<<endl;
        #endif

        vector<cv::Mat> *pFeatureImages;
		
        pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
        
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channelId];
    }

    virtual uint16_t getNbImages() const
    {
        return (uint16_t)vectSelectedImagesIndices.size();
    }

  public:
    vector<unsigned int> vectSelectedImagesIndices;
};


template <class FeatureType> class TrainingSetSelection2ndStage : public TrainingSet<FeatureType>
{
  public:
    TrainingSetSelection2ndStage(int nblab, ImageData *pData):TrainingSet<FeatureType>(nblab, pData)
    {
    }

    virtual FeatureType getValue(uint16_t imageId, uint8_t channel, int16_t x, int16_t y) const
    {
        ImageDataFloat2ndStageCombined *pData2ndStage = (ImageDataFloat2ndStageCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        if (channel>=0 && channel<pData1stStage->getNbFeatures())
        {
            vector<cv::Mat> *pFeatureImages;

            pFeatureImages = pData1stStage->getFeatureImages(vectSelectedImagesIndices[imageId]);
            assert(pFeatureImages!=NULL);

            return (*pFeatureImages)[channel].at<float>(y, x);
        }
        else {
            unsigned int iFeature2ndStage = channel - pData1stStage->getNbFeatures();
            unsigned int iScale, iLabel, iFeatureEntry;
            vector<cv::Mat> *pLabelIntegrals;
            cv::Mat *pImg_one, *pImg_x, *pImg_y, *pImg_xx, *pImg_yy, *pImg_xy;

            float featuresEntry[6];
            float &proportion = featuresEntry[0];
            float &mean_x = featuresEntry[1];
            float &mean_y = featuresEntry[2];
            float &var_x = featuresEntry[3];
            float &var_y = featuresEntry[4];
            float &cov_xy = featuresEntry[5];

            CLabelFeature::featureIndexToParams(iFeature2ndStage, iScale, iLabel, iFeatureEntry);

            pLabelIntegrals = pData2ndStage->getLabelIntegralImages(vectSelectedImagesIndices[imageId]);

            int bx, by, ex, ey;
            float nb_points;

            // cout<<"iFeatureIdx="<<iFeature2ndStage<<"  Scale="<<iScale<<"  label="<<iLabel<<"  iFeatureEntry="<<iFeatureEntry; //<<endl<<endl;

            bx = max(0, x-regionScales[iScale]/2-1);
            by = max(0, y-regionScales[iScale]/2-1);
            ex = min(this->iWidth-1, x+regionScales[iScale]/2);
            ey = min(this->iHeight-1, y+regionScales[iScale]/2);

            float wsize = (float)((ey-by)*(ex-bx));

            pImg_one = &((*pLabelIntegrals)[6*iLabel + 0]);
            pImg_x   = &((*pLabelIntegrals)[6*iLabel + 1]);
            pImg_y   = &((*pLabelIntegrals)[6*iLabel + 2]);
            pImg_xx  = &((*pLabelIntegrals)[6*iLabel + 3]);
            pImg_yy  = &((*pLabelIntegrals)[6*iLabel + 4]);
            pImg_xy  = &((*pLabelIntegrals)[6*iLabel + 5]);

            nb_points =
                pImg_one->at<float>(ey,ex) -
                pImg_one->at<float>(by,ex) -
                pImg_one->at<float>(ey,bx) +
                pImg_one->at<float>(by,bx);

            if (nb_points>=1.0f)
            {
                proportion = nb_points/wsize;

                mean_x = (float) (
                    pImg_x->at<float>(ey,ex) -
                    pImg_x->at<float>(by,ex) -
                    pImg_x->at<float>(ey,bx) +
                    pImg_x->at<float>(by,bx)
                    ) / nb_points;

                mean_y = (float) (
                    pImg_y->at<float>(ey,ex) -
                    pImg_y->at<float>(by,ex) -
                    pImg_y->at<float>(ey,bx) +
                    pImg_y->at<float>(by,bx)
                    ) / nb_points;

                var_x = (float) (
                    pImg_xx->at<float>(ey,ex) -
                    pImg_xx->at<float>(by,ex) -
                    pImg_xx->at<float>(ey,bx) +
                    pImg_xx->at<float>(by,bx)
                    ) / nb_points - mean_x*mean_x;

                var_y = (float) (
                    pImg_yy->at<float>(ey,ex) -
                    pImg_yy->at<float>(by,ex) -
                    pImg_yy->at<float>(ey,bx) +
                    pImg_yy->at<float>(by,bx)
                    ) / nb_points - mean_y*mean_y;

                cov_xy = (float) (
                    pImg_xy->at<float>(ey,ex) -
                    pImg_xy->at<float>(by,ex) -
                    pImg_xy->at<float>(ey,bx) +
                    pImg_xy->at<float>(by,bx)
                    ) / nb_points - mean_x*mean_y;

                mean_x -= (float)x;
                mean_y -= (float)y;
            }
            else {
                proportion = 0.0f;
                mean_x = FLT_MAX;
                mean_y = FLT_MAX;
                var_x = 0.0f;
                var_y = 0.0f;
                cov_xy = 0.0f;
            }
            // cout<<"  value="<<featuresEntry[iFeatureEntry]<<endl;
            return (FeatureType)featuresEntry[iFeatureEntry];
        }
    }

    // returns value from integral image representation (handling of +1 for pt2 coords must be done externally!)
    virtual FeatureType getValueIntegral(uint16_t imageId, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2) const
    {
        ImageDataFloat2ndStageCombined *pData2ndStage = (ImageDataFloat2ndStageCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        if (channel>=0 && channel<pData1stStage->getNbFeatures())
        {
            pData1stStage = pData2ndStage->GetData1stStage();

            if (pData1stStage->UseIntegralImages()==true)
            {
                vector<cv::Mat> *pFeatureImages;
                cv::Mat *pFeaturesIntegral;

                pFeatureImages = pData1stStage->getFeatureIntegralImages(vectSelectedImagesIndices[imageId]);
                assert(pFeatureImages!=NULL);

                pFeaturesIntegral = &((*pFeatureImages)[channel]);
                return pFeaturesIntegral->at<FeatureType>(y2, x2) -
                    pFeaturesIntegral->at<FeatureType>(y2, x1) -
                    pFeaturesIntegral->at<FeatureType>(y1, x2) +
                    pFeaturesIntegral->at<FeatureType>(y1, x1);
            }
            else {
                cout<<"ERROR in getValueIntegral(...): does not use integral images"<<endl;
                exit(-1);
                return (FeatureType)0;
            }
        }
        else {
            // cout<<"getValueIntegral : channel="<<channel<<" (x1,y1)=("<<x1<<","<<y1<<")  (x2,y2)=("<<x2<<","<<y2<<")"<<endl;

            unsigned int iFeature2ndStage = channel - pData1stStage->getNbFeatures();
            unsigned int iScale, iLabel, iFeatureEntry;
            vector<cv::Mat> *pLabelIntegrals;
            cv::Mat *pImg_one, *pImg_x, *pImg_y, *pImg_xx, *pImg_yy, *pImg_xy;

            float featuresEntry[6];
            float &proportion = featuresEntry[0];
            float &mean_x = featuresEntry[1];
            float &mean_y = featuresEntry[2];
            float &var_x = featuresEntry[3];
            float &var_y = featuresEntry[4];
            float &cov_xy = featuresEntry[5];

            CLabelFeature::featureIndexToParams(iFeature2ndStage, iScale, iLabel, iFeatureEntry);

            pLabelIntegrals = pData2ndStage->getLabelIntegralImages(vectSelectedImagesIndices[imageId]);

            int bx, by, ex, ey;
            float nb_points;

            int16_t x, y;
            FeatureType sum = (FeatureType)0;

            for (y=y1; y<=y2; y++)
            {
                for (x=x1; x<=x2; x++)
                {
                    bx = max(0, x-regionScales[iScale]/2-1);
                    by = max(0, y-regionScales[iScale]/2-1);
                    ex = min(this->iWidth-1, x+regionScales[iScale]/2);
                    ey = min(this->iHeight-1, y+regionScales[iScale]/2);

                    float wsize = (float)((ey-by)*(ex-bx));

                    pImg_one = &((*pLabelIntegrals)[6*iLabel + 0]);
                    pImg_x   = &((*pLabelIntegrals)[6*iLabel + 1]);
                    pImg_y   = &((*pLabelIntegrals)[6*iLabel + 2]);
                    pImg_xx  = &((*pLabelIntegrals)[6*iLabel + 3]);
                    pImg_yy  = &((*pLabelIntegrals)[6*iLabel + 4]);
                    pImg_xy  = &((*pLabelIntegrals)[6*iLabel + 5]);

                    nb_points =
                        pImg_one->at<float>(ey,ex) -
                        pImg_one->at<float>(by,ex) -
                        pImg_one->at<float>(ey,bx) +
                        pImg_one->at<float>(by,bx);

                    if (nb_points!=0.0f)
                    {
                        proportion = nb_points/wsize;

                        mean_x = (float) (
                            pImg_x->at<float>(ey,ex) -
                            pImg_x->at<float>(by,ex) -
                            pImg_x->at<float>(ey,bx) +
                            pImg_x->at<float>(by,bx)
                            ) / nb_points;

                        mean_y = (float) (
                            pImg_y->at<float>(ey,ex) -
                            pImg_y->at<float>(by,ex) -
                            pImg_y->at<float>(ey,bx) +
                            pImg_y->at<float>(by,bx)
                            ) / nb_points;

                        var_x = (float) (
                            pImg_xx->at<float>(ey,ex) -
                            pImg_xx->at<float>(by,ex) -
                            pImg_xx->at<float>(ey,bx) +
                            pImg_xx->at<float>(by,bx)
                            ) / nb_points - mean_x*mean_x;

                        var_y = (float) (
                            pImg_yy->at<float>(ey,ex) -
                            pImg_yy->at<float>(by,ex) -
                            pImg_yy->at<float>(ey,bx) +
                            pImg_yy->at<float>(by,bx)
                            ) / nb_points - mean_y*mean_y;

                        cov_xy = (float) (
                            pImg_xy->at<float>(ey,ex) -
                            pImg_xy->at<float>(by,ex) -
                            pImg_xy->at<float>(ey,bx) +
                            pImg_xy->at<float>(by,bx)
                            ) / nb_points - mean_x*mean_y;

                        mean_x -= (float)x;
                        mean_y -= (float)y;
                    }
                    else {
                        proportion = 0.0f;
                        mean_x = FLT_MAX;
                        mean_y = FLT_MAX;
                        var_x = 0.0f;
                        var_y = 0.0f;
                        cov_xy = 0.0f;
                    }
                    sum += (FeatureType)featuresEntry[iFeatureEntry];
                }
            }

            return sum;
        }
    }

    virtual uint8_t getLabel(uint16_t imageId, int16_t x, int16_t y) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabel(uint16_t, int16_t x 2)"<<endl;
        #endif
        ImageDataFloat2ndStageCombined *pData2ndStage = (ImageDataFloat2ndStageCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        cv::Mat *pLabelImage;

        pLabelImage = pData1stStage->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return pLabelImage->at<uint8_t>(y, x);
    }

    virtual cv::Mat getLabelImage(uint16_t imageId) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabelImage(uint16_t)"<<endl;
        #endif
        ImageDataFloat2ndStageCombined *pData2ndStage = (ImageDataFloat2ndStageCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        cv::Mat *pLabelImage;

        pLabelImage = pData1stStage->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return *pLabelImage;
    }

    virtual cv::Mat getImageData(uint16_t imageId, uint16_t channelId) const
    {
        cout<<"ERROR in TrainingSetSelection2ndStage::getImageData(): function deprecated"<<endl;
        exit(-1);
        return cv::Mat();
        /*
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getImageData(uint16_t, uint16_t)"<<endl;
        #endif

        vector<cv::Mat> *pFeatureImages;

        pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channelId];*/
    }

    virtual uint16_t getNbImages() const
    {
        return (uint16_t)vectSelectedImagesIndices.size();
    }

  public:
    vector<unsigned int> vectSelectedImagesIndices;
};


template <class FeatureType> class TrainingSetSelection2ndStageBinary : public TrainingSet<FeatureType>
{
  public:
    TrainingSetSelection2ndStageBinary(int nblab, ImageData *pData):TrainingSet<FeatureType>(nblab, pData)
    {
    }

    virtual FeatureType getValue(uint16_t imageId, uint8_t channel, int16_t x, int16_t y) const
    {
        ImageDataFloat2ndStageBinaryCombined *pData2ndStage = (ImageDataFloat2ndStageBinaryCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        if (channel>=0 && channel<pData1stStage->getNbFeatures())
        {
            vector<cv::Mat> *pFeatureImages;

            pFeatureImages = pData1stStage->getFeatureImages(vectSelectedImagesIndices[imageId]);
            assert(pFeatureImages!=NULL);

            return (*pFeatureImages)[channel].at<float>(y, x);
        }
        else {
            vector<cv::Mat> *pFeatureImages;

            pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
            assert(pFeatureImages!=NULL);

            return (*pFeatureImages)[channel - pData1stStage->getNbFeatures()].at<FeatureType>(y, x);
        }
    }

    // returns value from integral image representation (handling of +1 for pt2 coords must be done externally!)
    virtual FeatureType getValueIntegral(uint16_t imageId, uint8_t channel, int16_t x1, int16_t y1, int16_t x2, int16_t y2) const
    {
        ImageDataFloat2ndStageCombined *pData2ndStage = (ImageDataFloat2ndStageCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        if (channel>=0 && channel<pData1stStage->getNbFeatures())
        {
            pData1stStage = pData2ndStage->GetData1stStage();

            if (pData1stStage->UseIntegralImages()==true)
            {
                vector<cv::Mat> *pFeatureImages;
                cv::Mat *pFeaturesIntegral;

                pFeatureImages = pData1stStage->getFeatureIntegralImages(vectSelectedImagesIndices[imageId]);
                assert(pFeatureImages!=NULL);

                pFeaturesIntegral = &((*pFeatureImages)[channel]);
                return pFeaturesIntegral->at<FeatureType>(y2, x2) -
                    pFeaturesIntegral->at<FeatureType>(y2, x1) -
                    pFeaturesIntegral->at<FeatureType>(y1, x2) +
                    pFeaturesIntegral->at<FeatureType>(y1, x1);
            }
            else {
                cout<<"ERROR in getValueIntegral(...): does not use integral images"<<endl;
                exit(-1);
                return (FeatureType)0;
            }
        }
        else {
            if (pData2ndStage->UseIntegralImages()==true)
            {
                vector<cv::Mat> *pFeatureImages;
                cv::Mat *pFeaturesIntegral;

                pFeatureImages = pData2ndStage->getFeatureIntegralImages(vectSelectedImagesIndices[imageId]);
                assert(pFeatureImages!=NULL);

                pFeaturesIntegral = &((*pFeatureImages)[channel - pData1stStage->getNbFeatures()]);

                return pFeaturesIntegral->at<FeatureType>(y2, x2) -
                    pFeaturesIntegral->at<FeatureType>(y2, x1) -
                    pFeaturesIntegral->at<FeatureType>(y1, x2) +
                    pFeaturesIntegral->at<FeatureType>(y1, x1);
            }
            else {
                cout<<"ERROR in getValueIntegral(...): does not use integral images"<<endl;
                exit(-1);
                return (FeatureType)0;
            }
        }
    }

    virtual uint8_t getLabel(uint16_t imageId, int16_t x, int16_t y) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabel(uint16_t, int16_t x 2)"<<endl;
        #endif
        ImageDataFloat2ndStageBinaryCombined *pData2ndStage = (ImageDataFloat2ndStageBinaryCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        cv::Mat *pLabelImage;

        pLabelImage = pData1stStage->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return pLabelImage->at<uint8_t>(y, x);
    }

    virtual cv::Mat getLabelImage(uint16_t imageId) const
    {
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getLabelImage(uint16_t)"<<endl;
        #endif
        ImageDataFloat2ndStageBinaryCombined *pData2ndStage = (ImageDataFloat2ndStageBinaryCombined *)this->pImageData;
        ImageDataFloat *pData1stStage;

        pData1stStage = pData2ndStage->GetData1stStage();

        cv::Mat *pLabelImage;

        pLabelImage = pData1stStage->getLabelImage(vectSelectedImagesIndices[imageId]);
        assert(pLabelImage!=NULL);

        return *pLabelImage;
    }

    virtual cv::Mat getImageData(uint16_t imageId, uint16_t channelId) const
    {
        cout<<"ERROR in TrainingSetSelection2ndStage::getImageData(): function deprecated"<<endl;
        exit(-1);
        return cv::Mat();
        /*
        #ifdef DEEP_TRACE
        // cout<<"TrainingSet::getImageData(uint16_t, uint16_t)"<<endl;
        #endif

        vector<cv::Mat> *pFeatureImages;

        pFeatureImages = this->pImageData->getFeatureImages(vectSelectedImagesIndices[imageId]);
        assert(pFeatureImages!=NULL);

        return (*pFeatureImages)[channelId];*/
    }

    virtual uint16_t getNbImages() const
    {
        return (uint16_t)vectSelectedImagesIndices.size();
    }

  public:
    vector<unsigned int> vectSelectedImagesIndices;
};

}
#endif
