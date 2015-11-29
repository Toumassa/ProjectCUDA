#ifndef LABELFEATURE_H
#define LABELFEATURE_H

#include <opencv2/core/core.hpp>

// Labels start with zero
#define NO_REGION_SCALES	3
#define NO_LABELS			9
extern const int regFeatures;

typedef unsigned char labeltype;
//typedef unsigned int integraltype; //working
//typedef unsigned long long int integraltype;;

extern const int regionScales[3];

typedef enum  {
	FT_PROPORTION,
	FT_MX,
	FT_MY,
	FT_CXX,
	FT_CYY,
	FT_CXY,
	FT_ENTRYCOUNT // not used, gives the number of entries
} Features;

#define FTINDEX(SC,LAB,FT)	((SC)*(NO_LABELS)*(FT_ENTRYCOUNT)+(LAB)*(FT_ENTRYCOUNT)+(FT))


class IntImagesOneLabel {
public:
	cv::Mat/*_<integraltype>*/ x, y, xx, yy, xy, one;

	void create (int rows, int cols)
	{
		one.create(rows,cols, CV_32F);
		x.create(rows,cols, CV_32F);
		y.create(rows,cols, CV_32F);
		xx.create(rows,cols, CV_32F);
		yy.create(rows,cols, CV_32F);
		xy.create(rows,cols, CV_32F);
	}
};

class CLabelFeature
{
  // Member variables
  public:
    IntImagesOneLabel arrayIntegralImages[NO_LABELS];

  public:
	cv::Mat imgLabel;

  // Member functions
  private:
    void calcIntImages();
    void GetFeatures(const cv::Point &pt, int window_size, int label,
                     float &proportion, float &mean_x, float &mean_y, float &var_x, float &var_y, float &cov_xy);

    void GetFeaturesNaive(const cv::Point &pt, int window_size, int label,
                     float &proportion, float &mean_x, float &mean_y, float &var_x, float &var_y, float &cov_xy);

  public:
    static void featureIndexToParams(unsigned int iFeatureIdx,
    unsigned int &iScale, unsigned int &iLabel, unsigned int &iFeatureEntry)
    {
        iScale = iFeatureIdx/(NO_LABELS*FT_ENTRYCOUNT);
        iLabel = (iFeatureIdx%(NO_LABELS*FT_ENTRYCOUNT))/FT_ENTRYCOUNT;
        iFeatureEntry = iFeatureIdx - iScale*NO_LABELS*FT_ENTRYCOUNT - iLabel*FT_ENTRYCOUNT;
    }

    CLabelFeature() {}

    virtual void SetImage(const cv::Mat &);
    void MakeFeatureVector(const cv::Point &pt, float *);
};

#endif
