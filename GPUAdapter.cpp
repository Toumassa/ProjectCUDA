#include "GPUAdapter.h"



void GPUAdapter::treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node, int parent,int id, int* id_counter)
{
	 
     ANode anode;
     anode.id = id;
     anode.parent = parent;
     //node filling
		//TNode attributes
		 anode.depth=node->getDepth();
		 anode.start=node->getStart();
		 anode.end=node->getEnd();
		 anode.idx=node->getIdx();
		//Prediction attributes
		Prediction predict=node->getPrediction();
		anode.histSize=predict.n;

			//Adding node's hist vector to common vector with saving
			//           offset and size in nodes informations

		anode.common_hist_tab_offset = this->common_hist_tab.size();
		anode.common_hist_tab_size = predict.hist.size();
		for(int i =0; i < anode.common_hist_tab_size ;i++)
		{
			this->common_hist_tab.push_back(predict.hist[i]);
		}

			//Adding node's p vector to common vector with saving
			//           offset and size in nodes informations
		anode.common_p_tab_offset = this->common_p_tab.size();
		anode.common_p_tab_size = predict.p.size();
		
		for(int i =0; i < anode.common_p_tab_size;i++)
		{
			this->common_p_tab.push_back(predict.p[i]);
		}/**/


		anode.splitData = node->splitData;
		
     if(!(node->isLeaf()))
	 {
		 anode.left = ++(*id_counter);
		anode.right = ++(*id_counter);
	 }
	  else
	 {
		 anode.left = -1;
		 anode.right = -1;
	 }
	 arbre->push_back(anode);

	 if(!(node->isLeaf()))
	 {
		 
		 treeToVectorRecursif(arbre, node->getLeft(), anode.id,anode.left, id_counter);
		 treeToVectorRecursif(arbre, node->getRight(), anode.id,anode.right, id_counter);
	 }
} 

void GPUAdapter::treeToVector(vector<ANode> *treeAsVector, StrucClassSSF<float>*tree)
{
	int id=0;
	
    treeToVectorRecursif(treeAsVector, (*tree).root(), -1,0, &id);
}

/** Creates a new array representing the tree
 outputTree must be a pointer on 0x0

*/
/*void GPUAdapter::treeToTab(StrucClassSSF<float>*inputTree, ANode *outputTree, unsigned int *treeSize)
{
    vector<ANode> *treeVector = new vector<ANode>();
    
    treeToVector(treeVector, inputTree);
    *treeSize=treeVector->size();
    cout << "Vector Size: " << treeSize<<endl;
    
    outputTree = new ANode[*treeSize];
    for(int i = 0; i < *treeSize; i++){
        outputTree[i]=(*treeVector)[i];
       ANode node = outputTree[i];
       
       cout << " id: "<<  node.id << " parent: "<< node.parent << " left: "<< node.left << " right: " << node.right << endl;
    }
    
    delete treeVector;
}*/




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

    cout << "Vector Size: " << treeVector->size()<<endl;
    
    /*for(int i = 0; i < treeVector->size(); i++){
       //ANode node = (*treeVector)[i];
       
       //cout << " id: "<<  node.id << " parent: "<< node.parent << " left: "<< node.left << " right: " << node.right << endl;
    }*/

    this->treesAsVector.push_back(treeVector);
}

void GPUAdapter::SetTrainingSetSelection(TrainingSetSelection<float> *trainingset)
{
	this->ts = ts;

	this->iWidth = trainingset->getImgWidth(0);
	this->iHeight = trainingset->getImgHeight(0);
	this->nChannels = trainingset->getNChannels();

}


void* GPUAdapter::PushTreeToGPU(int n)
{
	if(n < 0 || n > this->treesAsVector.size())
	{
		cerr << "incorrect tree index" << endl;
		return NULL;
	}

	//change with malloc GPU
	ANode *treeAsTab = (ANode*)malloc( (this->treesAsVector[n])->size()*sizeof(ANode));

	return (void*)treeAsTab;
}



void GPUAdapter::getFlattenedFeatures(uint16_t imageId, float **out_features, int *out_nbChannels)
{
    vector<cv::Mat> *pFeatureImages = this->pImageData->getFeatureImages(this->ts->vectSelectedImagesIndices[imageId]);
    assert(pFeatureImages!=NULL);

    float *flat = (float *) malloc (sizeof(float)*(this->iWidth)*(this->iHeight)*(this->nChannels));
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
    *out_nbChannels = this->nChannels;
}

/***************************************************************************
	 For the GPU version: flatten all integral features in a single 1D table
	 ***************************************************************************/

void GPUAdapter::getFlattenedIntegralFeatures(uint16_t imageId, float **out_features_integral, int16_t *out_w, int16_t *out_h) 
{
    vector<cv::Mat> *pFeatureImages = this->pImageData->getFeatureIntegralImages(this->ts->vectSelectedImagesIndices[imageId]);
    assert(pFeatureImages!=NULL);
    assert(this->pImageData->UseIntegralImages()==true);

    int16_t w = (*pFeatureImages)[0].cols;
    int16_t h = (*pFeatureImages)[0].rows;
    float *flat = (float *) malloc (sizeof(float)*w*h*(this->nChannels));
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