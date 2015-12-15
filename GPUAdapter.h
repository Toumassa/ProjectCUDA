#ifndef DEF_GPU_ADAPTER
#define DEF_GPU_ADAPTER

#include <vector>
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
	void SetTrainingSetSelection(TrainingSetSelection<float> *ts);

	void test()
	{
		cout << "hist tab size : " << common_hist_tab.size() << endl;
		cout << "P tab size : " << common_p_tab.size() << endl;
	}


	void* PushTreeToGPU(int);
private:
	vector<vector<ANode>* > treesAsVector;


	vector<uint32_t> common_hist_tab; 
	vector<float> common_p_tab; 

	ImageData *pImageData;
	int iWidth, iHeight, nChannels;

	TrainingSetSelection<float> *ts;

	float *features;
	float *features_integral;

	void getFlattenedFeatures(uint16_t imageId, float **out_features, int *out_nbChannels);
	void getFlattenedIntegralFeatures(uint16_t imageId, float **out_features_integral, int16_t *out_w, int16_t *out_h);

	/*Private use functions for all trees*/
	void treeToVector(vector<ANode> *treeAsVector, StrucClassSSF<float>*tree);
	void treeToVectorRecursif(vector<ANode> *arbre, TNode<SplitData<float>, Prediction> *node, int parent,int id, int* id_counter);
};


#endif
