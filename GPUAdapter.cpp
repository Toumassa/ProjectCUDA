#include "GPUAdapter.h"
#include "kernel.h"

/**
	(RECURSIF)Converts tree structure to vector with storing hist data into another commun vector
*/
int GPUAdapter::treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node)
{
        ANode anode;
    
        //Prediction attributes
        Prediction predict=node->getPrediction();
        anode.histSize=predict.n;

        anode.common_hist_tab_offset = this->common_hist_tab.size();
        anode.common_hist_tab_size = predict.hist.size();
        int max = 0;
        int min = 0;
        for(int i =0; i < anode.common_hist_tab_size ;i++)
        {
            this->common_hist_tab.push_back(predict.hist[i]);
        }

        anode.splitData = node->splitData;
        anode.id = arbre->size();

        arbre->push_back(anode);

     if(!(node->isLeaf()))
     {
         int left = treeToVectorRecursif(arbre, node->getLeft());
         int right = treeToVectorRecursif(arbre, node->getRight());
         (*arbre)[anode.id].left = left;
         (*arbre)[anode.id].right = right;
     }
      else/**/
     {
         (*arbre)[anode.id].left = -1;
         (*arbre)[anode.id].right = -1;
     }

     return anode.id;
} 
/**
	Calls RECURSIF function with initialisation
*/
void GPUAdapter::treeToVector(vector<ANode> *treeAsVector, StrucClassSSF<float>*tree)
{
    treeToVectorRecursif(treeAsVector, (*tree).root());
}

GPUAdapter::~GPUAdapter()
{
	std::vector<vector<ANode>*>::iterator vi;
	for(vi = this->treesAsVector.begin(); vi != this->treesAsVector.end();vi++)
	{
		delete (*vi);
	}
}

/**
	Add new tree for prediction
*/
void GPUAdapter::AddTree(StrucClassSSF<float>*inputTree)
{
	vector<ANode> *treeVector = new vector<ANode>();

    treeToVector(treeVector, inputTree);
    this->treesAsVector.push_back(treeVector);
}


/***************************************************************************
	 For the GPU version: flatten all features in a single 1D table
***************************************************************************/
void GPUAdapter::getFlattenedFeatures(uint16_t imageId, float **out_features, uint16_t *out_nbChannels)
{
    vector<cv::Mat> *pFeatureImages = this->pImageData->getFeatureImages(this->ts->vectSelectedImagesIndices[imageId]);
    assert(pFeatureImages!=NULL);
	
	this->fSize =this->iWidth*this->iHeight*(this->nChannels);
    float *flat = (float *) malloc (sizeof(float)*fSize);
    if (flat==NULL)
    {
    	std::cerr << "Cannot allocate flat feature data\n";
    	exit(1);
    }
    
    for (int c=0; c<this->nChannels; ++c)
    for (int x=0; x<this->iWidth; ++x)
    for (int y=0; y<this->iHeight; ++y)
    	flat[y+x*(this->iHeight)+c*(this->iHeight)*(this->iWidth)] =       
    		(*pFeatureImages)[c].at<float>(y, x);
    
    *out_features = flat;
}

/***************************************************************************
	 For the GPU version: flatten all integral features in a single 1D table
***************************************************************************/

void GPUAdapter::getFlattenedIntegralFeatures(uint16_t imageId, float **out_features_integral, uint16_t *out_w, uint16_t *out_h) 
{
    vector<cv::Mat> *pFeatureImages = this->pImageData->getFeatureIntegralImages(this->ts->vectSelectedImagesIndices[imageId]);
    assert(pFeatureImages!=NULL);
    assert(this->pImageData->UseIntegralImages()==true);

    int16_t w = (*pFeatureImages)[0].cols;
    int16_t h = (*pFeatureImages)[0].rows;
    
    this->fIntegralSize = w*h*(this->nChannels);
    float *flat = (float *) malloc ((int)sizeof(float)*fIntegralSize);
    if (flat==NULL)
    {
    	std::cerr << "Cannot allocate flat integral feature data\n";
    	exit(1);
    }
    

    for (int c=0; c<this->nChannels; ++c)
    for (int x=0; x<w; ++x)
    for (int y=0; y<h; ++y)
    	flat[y+x*h+c*h*w]  =
    		(*pFeatureImages)[c].at<float>(y, x);
    
    *out_w = w;
    *out_h = h;
    *out_features_integral = flat;
}

