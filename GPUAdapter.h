#ifndef DEF_GPU_ADAPTER
#define DEF_GPU_ADAPTER

#include <vector>
#include <iostream>
#include "SemanticSegmentationForests.h"
#include "StrucClassSSF.h"

using namespace std;
using namespace vision;

typedef struct ANode{
	//TNode
	int  id;
	int parent;
	int left;
	int right;
	
    ///histSize and hist storage
    uint32_t histSize;
    int  common_hist_tab_offset;
    int  common_hist_tab_size; // not used

	SplitData<float> splitData;


} ANode;

class GPUAdapter
{
public:
	GPUAdapter(){}
	~GPUAdapter();
	void AddTree(StrucClassSSF<float>*tree);

	void testGPUSolution(cv::Rect, Sample<float> &s);
	
	void init(StrucClassSSF<float> *forest, ConfigReader *cr);
	
	void preKernel(uint16_t imageId, ConfigReader *cr, TrainingSetSelection<float> *pTS);
	void postKernel(cv::Mat*);
	void PushTreeToGPU(int);
	
	void destroy();
private:
	vector<vector<ANode>* > treesAsVector;
	ANode **treeAsTab;
	vector<ANode*> _treeAsTab;
	unsigned int treeTabCount;

	float *features;
	float *features_integral;
	
	float *_features;
	float *_features_integral;
	
	int fSize;
	int fIntegralSize;
	
	int *result;
	int *resultGPU;
	

	vector<uint32_t> common_hist_tab; 
	
	uint32_t *_common_hist_tab;
	//vector<float> common_p_tab; 

	ImageData *pImageData;
	uint16_t iWidth, iHeight, nChannels, numLabels;
	uint16_t w_integral, h_integral;

	TrainingSetSelection<float> *ts;

	int lPXOff;
	int lPYOff;


	void getFlattenedFeatures(uint16_t imageId, float **out_features, uint16_t *out_nbChannels);
	void getFlattenedIntegralFeatures(uint16_t imageId, float **out_features_integral, uint16_t *out_w, uint16_t *out_h);

	/*Private use functions for all trees*/
	void treeToVector(vector<ANode> *treeAsVector, StrucClassSSF<float>*tree);
	//void treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node, int parent,int id, int* id_counter);
	int treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node);
};


#endif
