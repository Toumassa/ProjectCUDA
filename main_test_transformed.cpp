// =========================================================================================
// 
// =========================================================================================
//    Original message from Peter Kontschieder and Samuel Rota Bulò:
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

#include <iostream>
#include <unistd.h>
#include <omp.h>
#include <sys/stat.h>

#include "Global.h"
#include "ConfigReader.h"
#include "ImageData.h"
#include "ImageDataFloat.h"
#include "SemanticSegmentationForests.h"
#include "StrucClassSSF.h"


#include "label.h"
#include "GPUAdapter.h"

using namespace std;
using namespace vision;




/***************************************************************************
 USAGE
 ***************************************************************************/

void usage (char *com) 
{
    std::cerr<< "usage: " << com << " <configfile> <inputimage> <outputimage> <n.o.trees> <tree-model-prefix> <[0|1] -> CPU | GPU> <repeatCount>\n"
        ;
    exit(1);
}

/***************************************************************************
 Writes profiling output (milli-seconds since last call)
 ***************************************************************************/

clock_t LastProfilingClock;

inline float profiling (const char *s, clock_t *whichClock=NULL) 
{
	if (whichClock==NULL)
		whichClock=&LastProfilingClock;

    clock_t newClock=clock();
    float res = (float) (newClock-*whichClock) / (float) CLOCKS_PER_SEC;
    if (s!=NULL)
        std::cerr << "Time: " << s << ": " << res << std::endl; 
    *whichClock = newClock;
    return res;
}

inline float profilingTime (const char *s, time_t *whichClock) 
{
    time_t newTime=time(NULL);
    float res = (float) (newTime-*whichClock);
    if (s!=NULL)
        std::cerr << "Time(real): " << s << ": " << res << std::endl; 
    return res;
}

/***************************************************************************
 Test a simple image
 ***************************************************************************/



void testStructClassForest(StrucClassSSF<float> *forest, ConfigReader *cr, TrainingSetSelection<float> *pTS, int execMode, int imageCount)
{
    GPUAdapter newGPUAdapter;
    int iImage;
    cv::Point pt;
    cv::Mat matConfusion;
    char strOutput[200];
    
	
	if(execMode ==1)
	{
		newGPUAdapter.init(forest, cr);
		profiling("Init");
	}
    /**/
    // Process all test images
    // result goes into ====> result[].at<>(pt)
	
	if(execMode ==1)
	{
		
	}
	else
	{
		profiling("");
	}
	int count;
    for (iImage = 0, count= 0; iImage < pTS->getNbImages()&&count <imageCount; ++count)
    {
    	// Create a sample object, which contains the imageId
        Sample<float> s;

        s.imageId = iImage;
        cv::Rect box(0, 0, pTS->getImgWidth(s.imageId), pTS->getImgHeight(s.imageId));
        cv::Mat mapResult = cv::Mat::ones(box.size(), CV_8UC1) * cr->numLabels;
		
		if(execMode == 1)
		{
			//profiling("");
			newGPUAdapter.preKernel(s.imageId, cr, pTS);
			//profiling("PreKernel");
			newGPUAdapter.testGPUSolution(box, s);
			newGPUAdapter.postKernel(&mapResult);
			//profiling("[!]  Prediction ");
			//profiling("PostKernel");
		}
		else
		{
			int lPXOff = cr->labelPatchWidth / 2;
			int lPYOff = cr->labelPatchHeight / 2;

			// Initialize the result matrices
			vector<cv::Mat> result(cr->numLabels);
			for(int j = 0; j < result.size(); ++j)
				result[j] = Mat::zeros(box.size(), CV_32FC1);
			
			// Iterate over input image pixels
			for(s.y = 0; s.y < box.height; ++s.y)
			for(s.x = 0; s.x < box.width; ++s.x)
			{
				// Obtain forest predictions
				// Iterate over all trees
				for(size_t t = 0; t < cr->numTrees; ++t)
				{
					// The prediction itself.
					// The given Sample object s contains the imageId and the pixel coordinates.
					// p is an iterator to a vector over labels (attribut hist of class Prediction)
					// This labels correspond to a patch centered on position s
					// (this is the structured version of a random forest!)
					vector<uint32_t>::const_iterator p = forest[t].predictPtr(s);


					for (pt.y=(int)s.y-lPYOff;pt.y<=(int)s.y+(int)lPYOff;++pt.y)
					for (pt.x=(int)s.x-(int)lPXOff;pt.x<=(int)s.x+(int)lPXOff;++pt.x,++p)
					{
						if (*p<0 || *p >= (size_t)cr->numLabels)
						{
							std::cerr << "Invalid label in prediction: " << (int) *p << "\n";
							exit(1);
						}

						if (box.contains(pt))
						{
							result[*p].at<float>(pt) += 1;

						}
					}

				}
			}

			// Argmax of result ===> mapResult
			size_t maxIdx;
			for (pt.y = 0; pt.y < box.height; ++pt.y)
			for (pt.x = 0; pt.x < box.width; ++pt.x)
			{
				maxIdx = 0;


				for(int j = 1; j < cr->numLabels; ++j)
				{

					maxIdx = (result[j].at<float>(pt) > result[maxIdx].at<float>(pt)) ? j : maxIdx;
				}

				mapResult.at<uint8_t>(pt) = (uint8_t)maxIdx;
			}
		}
		
        // Write segmentation map
        sprintf(strOutput, "%s/segmap_1st_stage%04d.png", cr->outputFolder.c_str(), iImage);
        if (cv::imwrite(strOutput, mapResult)==false)
        {
            cout<<"Failed to write to "<<strOutput<<endl;
            return;
        }

        // Write RGB segmentation map
        cv::Mat imgResultRGB;
        convertLabelToRGB(mapResult, imgResultRGB);

        sprintf(strOutput, "%s/segmap_1st_stage_RGB%04d.png", cr->outputFolder.c_str(), iImage);
        if (cv::imwrite(strOutput, imgResultRGB)==false)
        {
            cout<<"Failed to write to "<<strOutput<<endl;
            return;
        } 
    }    
	if(execMode ==1)
	{
		
	}
	else
	{
		
	}
	profiling("[!] Total Prediction ");
	if(execMode == 1)
	{
		newGPUAdapter.destroy();
		profiling("Destroy");
	}

}

/***************************************************************************
 MAIN PROGRAM
 ***************************************************************************/

int main(int argc, char* argv[])
{
    string strConfigFile;
    ConfigReader cr;
    ImageData *idata = new ImageDataFloat();
    TrainingSetSelection<float> *pTrainingSet;
    bool bTestAll = false;
    int optNumTrees=-1, executionType=0, imageCount=0;
    char *optTreeFnamePrefix=NULL;
    char buffer[2048];

    srand(time(0));
    setlocale(LC_NUMERIC, "C");
    profiling(NULL);

#ifndef NDEBUG
    std::cout << "******************************************************\n"
    	<< "DEBUG MODE!!!!!\n"
		<< "******************************************************\n";
#endif

    if (argc!=6)
        usage(*argv);
    else
    {
        strConfigFile = argv[1];
        optNumTrees = atoi(argv[2]);
        optTreeFnamePrefix = argv[3];
        executionType = atoi(argv[4]);
        imageCount = atoi(argv[5]);
    }

    if (cr.readConfigFile(strConfigFile)==false)
    {
        cout<<"Failed to read config file "<<strConfigFile<<endl;
        return -1;
    }

    // Load image data
    idata->bGenerateFeatures = true;

    if (idata->setConfiguration(cr)==false)
    {
        cout<<"Failed to initialize image data with configuration"<<endl;
        return -1;
    }

    if (bTestAll==true)
    {
        std::cout << "Set contains all images. Not supported.\n";
        exit(1);
    }
    else {
        
        // CW Create a dummy training set selection with a single image number
        pTrainingSet = new TrainingSetSelection<float>(9, idata);
        ((TrainingSetSelection<float> *)pTrainingSet)->vectSelectedImagesIndices.push_back(0);
    }

    cout<<pTrainingSet->getNbImages()<<" test images"<<endl;

    // Load forest
    StrucClassSSF<float> *forest = new StrucClassSSF<float>[optNumTrees];

    profiling("Init + feature extraction");

    cr.numTrees = optNumTrees;
    cout << "Loading " << cr.numTrees << " trees: \n";

    for(int iTree = 0; iTree < optNumTrees; ++iTree)
    {
        sprintf(buffer, "%s%d.txt", optTreeFnamePrefix, iTree+1);
        std::cout << "Loading tree from file " << buffer << "\n";

        forest[iTree].bUseRandomBoxes = true;
        forest[iTree].load(buffer);
        forest[iTree].setTrainingSet(pTrainingSet);
    }
    cout << "done!" << endl;
    profiling("Tree loading");
    
	int execMode = 0;
	
	if(executionType==1)
		execMode = 1;
	
    testStructClassForest(forest, &cr, pTrainingSet, execMode, imageCount);

    // delete tree;
    delete pTrainingSet;
	delete idata;
    delete [] forest;


    std::cout << "Terminated successfully.\n";

    return 0;
}
