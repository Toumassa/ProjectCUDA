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
	uint32_t start, end;
	uint16_t depth;
	uint32_t idx;
	
	//Prediction
	/*uint32_t histSize;
	uint32_t* hist;
	float* p;*/

    ///histSize and hist storage
    uint32_t histSize;
    int  common_hist_tab_offset;
    int  common_hist_tab_size;

    // p
    int   common_p_tab_offset;
    int   common_p_tab_size;



	//SplitData can be stored as it is
	SplitData<float> splitData;

	/*int16_t dx1, dx2;
	int16_t dy1, dy2;
	int8_t bw1, bh1, bw2, bh2;
	uint8_t channel0;    // number of feature channels is restricted by 255
	uint8_t channel1;
	uint8_t fType;           // CW: split type
	/*FeatureType*/ /*float thres; */
	
	///
	/*inline bool isLeaf()
	{
		return ((this->left == -1)/);
	}*/

} ANode;

class GPUAdapter
{
public:
	GPUAdapter(){}
	~GPUAdapter();
	void AddTree(StrucClassSSF<float>*tree);

	void testCPUSolution(cv::Mat*, cv::Rect, Sample<float> &s);
	void testGPUSolution(cv::Rect, Sample<float> &s);
	
	void init(StrucClassSSF<float> *forest, ConfigReader *cr);
	
	void preKernel(uint16_t imageId, ConfigReader *cr, TrainingSetSelection<float> *pTS);
	void postKernel(cv::Mat*);
	ANode* PushTreeToCPU(int);
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
