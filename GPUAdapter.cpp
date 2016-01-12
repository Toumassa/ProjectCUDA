#include "GPUAdapter.h"
#include "kernel.h"

int GPUAdapter::treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node)
{
        ANode anode;
    
        //TNode attributes
         anode.depth=node->getDepth();
         anode.start=node->getStart();
         anode.end=node->getEnd();
         anode.idx=node->getIdx();
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
void GPUAdapter::treeToVector(vector<ANode> *treeAsVector, StrucClassSSF<float>*tree)
{
	int id=0;
	
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

void GPUAdapter::AddTree(StrucClassSSF<float>*inputTree)
{
	vector<ANode> *treeVector = new vector<ANode>();

    treeToVector(treeVector, inputTree);
    this->treesAsVector.push_back(treeVector);
}


ANode* GPUAdapter::PushTreeToCPU(int n)
{
	if(n < 0 || n > this->treesAsVector.size())
	{
		cerr << "incorrect tree index" << endl;
		return NULL;
	}

	//change with malloc GPU
	ANode *treeAsTab = (ANode*)malloc( (this->treesAsVector[n])->size()*sizeof(ANode));
	for(int i = 0; i < (this->treesAsVector[n])->size();i++)
	{
		treeAsTab[i] = (*(this->treesAsVector[n]))[i];
	}		
	return treeAsTab;
}



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

void GPUAdapter::testCPUSolution(cv::Mat*mapResult, cv::Rect box, Sample<float>&s)
{
    /*int returnStartHistTab, returnCountHistTab;

    cv::Point pt;

	s.x = 0;
	s.y = 0;

        // Initialize the result matrices
    vector<cv::Mat> result(this->numLabels);
    for(int j = 0; j < result.size(); ++j)
        result[j] = Mat::zeros(box.size(), CV_32FC1);

    
    for(size_t t = 0; t < this->treeTabCount; ++t)
    {
    // Iterate over input image pixels
        for(s.y = 0; s.y < box.height; ++s.y)
        for(s.x = 0; s.x < box.width; ++s.x)
        {
        // Obtain forest predictions
        // Iterate over all trees
        
        	// The prediction itself.
        	// The given Sample object s contains the imageId and the pixel coordinates.
            // p is an iterator to a vector over labels (attribut hist of class Prediction)
            // This labels correspond to a patch centered on position s
            // (this is the structured version of a random forest!)
           // vector<uint32_t>::const_iterator p = forest[t].predictPtr(s);

            predict(&returnStartHistTab, &returnCountHistTab, this->treeAsTab[t], this->iWidth, this->iHeight, this->w_integral, this->h_integral, this->features, this->features_integral, s);
            int p = returnStartHistTab;
            

           // cout << "p : " << p << endl;
            for (pt.y=(int)s.y-this->lPYOff;pt.y<=(int)s.y+(int)this->lPYOff;++pt.y)
            for (pt.x=(int)s.x-(int)this->lPXOff;pt.x<=(int)s.x+(int)this->lPXOff;++pt.x,++p)
            {
            	   if (this->common_hist_tab[p]<0 || this->common_hist_tab[p] >= (size_t)this->numLabels)
                    {
                        cout << "x:" << s.x << " y:"<<s.y << " tree:"<< t << endl;
                        cout << "pt.x:" << pt.x << " pt.y:"<<pt.y << ":"<< p << endl;
                        cout << "*p : " << this->common_hist_tab[p] << endl;
                        std::cerr << "Invalid label in prediction: " << (int) this->common_hist_tab[p] << "\n";
                        exit(1);
                    }         
                if (box.contains(pt))
				{	
					
                    result[this->common_hist_tab[p]].at<float>(pt) += 1;
                    //result[*p].at<float>(pt) += 1;
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


        for(int j = 1; j < this->numLabels; ++j)
        {

            maxIdx = (result[j].at<float>(pt) > result[maxIdx].at<float>(pt)) ? j : maxIdx;
        }

        (*mapResult).at<uint8_t>(pt) = (uint8_t)maxIdx;
    }
*/

}
