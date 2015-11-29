#include <iostream>
#include "labelfeature.h"

using namespace std;

const int regFeatures = NO_LABELS * 6* NO_REGION_SCALES;
const int regionScales[3] = {15, 29, 45};

void CLabelFeature::calcIntImages()
{
	// SPECIAL CASE: FIRST PIXEL (UPPER LEFT CORNER)
	for (int l=0; l<NO_LABELS; ++l)
	{
	    arrayIntegralImages[l].create(imgLabel.rows, imgLabel.cols);

		arrayIntegralImages[l].one.at<float>(0,0) = 0;
		arrayIntegralImages[l].x.at<float>(0,0) = 0;
		arrayIntegralImages[l].y.at<float>(0,0) = 0;
		arrayIntegralImages[l].xx.at<float>(0,0) = 0;
		arrayIntegralImages[l].yy.at<float>(0,0) = 0;
		arrayIntegralImages[l].xy.at<float>(0,0) = 0;
	}

	labeltype curl = imgLabel.at<labeltype>(0, 0);

	if (curl>=0 && curl<NO_LABELS)
	{
	}
	else cout<<"WRONG LABEL VALUE"<<endl;
	//unsigned char curl = lab.at<labeltype>(0, 0);

	arrayIntegralImages[curl].one.at<float>(0,0) = 1;


	// SPECIAL CASE: FIRST LINE (first row)
	for (int x=1; x<imgLabel.cols; ++x)
	{
		curl = imgLabel.at<labeltype>(0, x);

		if (curl>=0 && curl<NO_LABELS)
        {
        }
        else cout<<"WRONG LABEL VALUE"<<endl;

		for (int l=0; l<NO_LABELS; ++l)
		{
			arrayIntegralImages[l].one.at<float>(0,x) =
				arrayIntegralImages[l].one.at<float>(0,x-1);
			// cout<<arrayIntegralImages[l].one.at<float>(0,x)<<" ";
			arrayIntegralImages[l].x.at<float>(0,x) =
				arrayIntegralImages[l].x.at<float>(0,x-1);

			arrayIntegralImages[l].y.at<float>(0,x) =
				arrayIntegralImages[l].y.at<float>(0,x-1);

			arrayIntegralImages[l].xx.at<float>(0,x) =
				arrayIntegralImages[l].xx.at<float>(0,x-1);

			arrayIntegralImages[l].yy.at<float>(0,x) =
				arrayIntegralImages[l].yy.at<float>(0,x-1);

			arrayIntegralImages[l].xy.at<float>(0,x) =
				arrayIntegralImages[l].xy.at<float>(0,x-1);
		}

		arrayIntegralImages[curl].one.at<float>(0,x) += 1;
		arrayIntegralImages[curl].x.at<float>(0,x) += x;
		arrayIntegralImages[curl].xx.at<float>(0,x) += x*x;
	}

	// SPECIAL CASE: FIRST COLUMN
	for (int y=1; y<imgLabel.rows; ++y) {

		curl = imgLabel.at<labeltype>(y, 0);
		if (curl>=0 && curl<NO_LABELS)
        {
        }
        else cout<<"WRONG LABEL VALUE"<<endl;

		for (int ll=0; ll<NO_LABELS; ++ll)
		{
			arrayIntegralImages[ll].one.at<float>(y,0) =
				arrayIntegralImages[ll].one.at<float>(y-1,0);

			arrayIntegralImages[ll].x.at<float>(y,0) =
				arrayIntegralImages[ll].x.at<float>(y-1,0);

			arrayIntegralImages[ll].y.at<float>(y,0) =
				arrayIntegralImages[ll].y.at<float>(y-1,0);

			arrayIntegralImages[ll].xx.at<float>(y,0) =
				arrayIntegralImages[ll].xx.at<float>(y-1,0);

			arrayIntegralImages[ll].yy.at<float>(y,0) =
				arrayIntegralImages[ll].yy.at<float>(y-1,0);

			arrayIntegralImages[ll].xy.at<float>(y,0) =
				arrayIntegralImages[ll].xy.at<float>(y-1,0);
		}


		arrayIntegralImages[curl].one.at<float>(y,0) += 1;
		arrayIntegralImages[curl].y.at<float>(y,0) += y;
		arrayIntegralImages[curl].yy.at<float>(y,0) += y*y;
		// all the other values are zero, since x=0;
	}


	// GENERAL CASE (NO BORDERS)
	for (int y=1; y<imgLabel.rows; ++y)
	{
		for (int x=1; x<imgLabel.cols; ++x)
		{
			curl = imgLabel.at<labeltype>(y, x);

			for (int l=0; l<NO_LABELS; ++l)
			{
				arrayIntegralImages[l].one.at<float>(y,x) =
					arrayIntegralImages[l].one.at<float>(y-1,x)+
					arrayIntegralImages[l].one.at<float>(y,x-1)-
					arrayIntegralImages[l].one.at<float>(y-1,x-1);

				arrayIntegralImages[l].x.at<float>(y,x) =
					arrayIntegralImages[l].x.at<float>(y-1,x)+
					arrayIntegralImages[l].x.at<float>(y,x-1)-
					arrayIntegralImages[l].x.at<float>(y-1,x-1);

				arrayIntegralImages[l].y.at<float>(y,x) =
					arrayIntegralImages[l].y.at<float>(y-1,x)+
					arrayIntegralImages[l].y.at<float>(y,x-1)-
					arrayIntegralImages[l].y.at<float>(y-1,x-1);

				arrayIntegralImages[l].xx.at<float>(y,x) =
					arrayIntegralImages[l].xx.at<float>(y-1,x)+
					arrayIntegralImages[l].xx.at<float>(y,x-1)-
					arrayIntegralImages[l].xx.at<float>(y-1,x-1);

				arrayIntegralImages[l].yy.at<float>(y,x) =
					arrayIntegralImages[l].yy.at<float>(y-1,x)+
					arrayIntegralImages[l].yy.at<float>(y,x-1)-
					arrayIntegralImages[l].yy.at<float>(y-1,x-1);

				arrayIntegralImages[l].xy.at<float>(y,x) =
					arrayIntegralImages[l].xy.at<float>(y-1,x)+
					arrayIntegralImages[l].xy.at<float>(y,x-1)-
					arrayIntegralImages[l].xy.at<float>(y-1,x-1);
			}

			arrayIntegralImages[curl].one.at<float>(y,x) += 1;
			arrayIntegralImages[curl].x.at<float>(y,x) += x;
			arrayIntegralImages[curl].y.at<float>(y,x) += y;
			arrayIntegralImages[curl].xx.at<float>(y,x) += x*x;
			arrayIntegralImages[curl].yy.at<float>(y,x) += y*y;
			arrayIntegralImages[curl].xy.at<float>(y,x) += x*y;
		}
	}
}

void CLabelFeature::GetFeatures(const cv::Point &pt, int window_size, int label,
                                   float &proportion, float &mean_x, float &mean_y, float &var_x, float &var_y, float &cov_xy)
{
    int bx, by, ex, ey;
    int x, y;
    float nb_points;

    x = pt.x;
    y = pt.y;

    bx = max(0, x-window_size/2-1),
    by = max(0, y-window_size/2-1),
    ex = min(imgLabel.cols-1, x+window_size/2),
    ey = min(imgLabel.rows-1, y+window_size/2);

    float wsize = (float)((ey-by)*(ex-bx));

    nb_points = (float) (
        arrayIntegralImages[label].one.at<float>(ey,ex) -
        arrayIntegralImages[label].one.at<float>(by,ex) -
        arrayIntegralImages[label].one.at<float>(ey,bx) +
        arrayIntegralImages[label].one.at<float>(by,bx));

    if (nb_points!=0.0f)
    {
        proportion = nb_points/wsize;

        mean_x = (float) (
            arrayIntegralImages[label].x.at<float>(ey,ex) -
            arrayIntegralImages[label].x.at<float>(by,ex) -
            arrayIntegralImages[label].x.at<float>(ey,bx) +
            arrayIntegralImages[label].x.at<float>(by,bx)
            ) / nb_points;

        mean_y = (float) (
            arrayIntegralImages[label].y.at<float>(ey,ex) -
            arrayIntegralImages[label].y.at<float>(by,ex) -
            arrayIntegralImages[label].y.at<float>(ey,bx) +
            arrayIntegralImages[label].y.at<float>(by,bx)
            ) / nb_points;

        var_x = (float) (
            arrayIntegralImages[label].xx.at<float>(ey,ex) -
            arrayIntegralImages[label].xx.at<float>(by,ex) -
            arrayIntegralImages[label].xx.at<float>(ey,bx) +
            arrayIntegralImages[label].xx.at<float>(by,bx)
            ) / nb_points - mean_x*mean_x;

        var_y = (float) (
            arrayIntegralImages[label].yy.at<float>(ey,ex) -
            arrayIntegralImages[label].yy.at<float>(by,ex) -
            arrayIntegralImages[label].yy.at<float>(ey,bx) +
            arrayIntegralImages[label].yy.at<float>(by,bx)
            ) / nb_points - mean_y*mean_y;

        cov_xy = (float) (
            arrayIntegralImages[label].xy.at<float>(ey,ex) -
            arrayIntegralImages[label].xy.at<float>(by,ex) -
            arrayIntegralImages[label].xy.at<float>(ey,bx) +
            arrayIntegralImages[label].xy.at<float>(by,bx)
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
}


void CLabelFeature::GetFeaturesNaive(const cv::Point &pt, int window_size, int label,
                                   float &proportion, float &mean_x, float &mean_y, float &var_x, float &var_y, float &cov_xy)
{
    int bx, by, ex, ey;
    int x, y;
    float nb_points;
    unsigned char ucLabel = (unsigned char)label;

    bx = max(0, pt.x-window_size/2),
    by = max(0, pt.y-window_size/2),
    ex = min(imgLabel.cols-1, pt.x+window_size/2),
    ey = min(imgLabel.rows-1, pt.y+window_size/2);

    float wsize = (float)((ey-by+1)*(ex-bx+1));
    float sum_x, sum_y;

    nb_points = 0.0f;
    sum_x = 0.0f;
    sum_y = 0.0f;
    for (y=by; y<=ey; y++)
    {
        for (x=bx; x<=ex; x++)
        {
            if (imgLabel.at<unsigned char>(y,x)==ucLabel)
            {
                nb_points+=1.0f;
                sum_x += x-pt.x;
                sum_y += y-pt.y;
            }
        }
    }

    if (nb_points!=0.0f)
    {
        proportion = nb_points/wsize;

        mean_x = sum_x/nb_points;
        mean_y = sum_y/nb_points;
        var_x = 0.0f;
        var_y = 0.0f;
        cov_xy = 0.0f;

        for (y=by; y<=ey; y++)
        {
            for (x=bx; x<=ex; x++)
            {
                if (imgLabel.at<unsigned char>(y,x)==ucLabel)
                {
                    var_x += ((x-pt.x)-mean_x)*((x-pt.x)-mean_x);
                    var_y += ((y-pt.y)-mean_y)*((y-pt.y)-mean_y);
                    cov_xy += ((x-pt.x)-mean_x)*((y-pt.y)-mean_y);
                }
            }
        }
        var_x /= nb_points;
        var_y /= nb_points;
        cov_xy /= nb_points;
    }
    else {
        proportion = 0.0f;
        mean_x = 0.0f;
        mean_y = 0.0f;
        var_x = 0.0f;
        var_y = 0.0f;
        cov_xy = 0.0f;
    }
}

void CLabelFeature::MakeFeatureVector(const cv::Point &pt, float *features)
{
    int bx, by, ex, ey;
    int x, y;
    float nb_points, proportion, mean_x, mean_y, var_x, var_y, cov_xy;
    int label, scale;

    x = pt.x;
    y = pt.y;

    float wsize;

    for (scale=0; scale<NO_REGION_SCALES; scale++)
    {
        bx = max(0, x-regionScales[scale]/2-1),
        by = max(0, y-regionScales[scale]/2-1),
        ex = min(imgLabel.cols-1, x+regionScales[scale]/2),
        ey = min(imgLabel.rows-1, y+regionScales[scale]/2);

        for (label=0; label<NO_LABELS; label++)
        {
            nb_points = (float) (
                arrayIntegralImages[label].one.at<float>(ey,ex) -
                arrayIntegralImages[label].one.at<float>(by,ex) -
                arrayIntegralImages[label].one.at<float>(ey,bx) +
                arrayIntegralImages[label].one.at<float>(by,bx));

            if (nb_points!=0.0f)
            {
                wsize = (float)((ey-by)*(ex-bx));
                proportion = nb_points/wsize;

                mean_x = (float) (
                    arrayIntegralImages[label].x.at<float>(ey,ex) -
                    arrayIntegralImages[label].x.at<float>(by,ex) -
                    arrayIntegralImages[label].x.at<float>(ey,bx) +
                    arrayIntegralImages[label].x.at<float>(by,bx)
                    ) / nb_points;

                mean_y = (float) (
                    arrayIntegralImages[label].y.at<float>(ey,ex) -
                    arrayIntegralImages[label].y.at<float>(by,ex) -
                    arrayIntegralImages[label].y.at<float>(ey,bx) +
                    arrayIntegralImages[label].y.at<float>(by,bx)
                    ) / nb_points;

                var_x = (float) (
                    arrayIntegralImages[label].xx.at<float>(ey,ex) -
                    arrayIntegralImages[label].xx.at<float>(by,ex) -
                    arrayIntegralImages[label].xx.at<float>(ey,bx) +
                    arrayIntegralImages[label].xx.at<float>(by,bx)
                    ) / nb_points - mean_x*mean_x;

                var_y = (float) (
                    arrayIntegralImages[label].yy.at<float>(ey,ex) -
                    arrayIntegralImages[label].yy.at<float>(by,ex) -
                    arrayIntegralImages[label].yy.at<float>(ey,bx) +
                    arrayIntegralImages[label].yy.at<float>(by,bx)
                    ) / nb_points - mean_y*mean_y;

                cov_xy = (float) (
                    arrayIntegralImages[label].xy.at<float>(ey,ex) -
                    arrayIntegralImages[label].xy.at<float>(by,ex) -
                    arrayIntegralImages[label].xy.at<float>(ey,bx) +
                    arrayIntegralImages[label].xy.at<float>(by,bx)
                    ) / nb_points - mean_x*mean_y;

                mean_x -= (float)x;
                mean_y -= (float)y;
            }
            else {
                proportion = 0.0f;
                mean_x = 0.0f;
                mean_y = 0.0f;
                var_x = 0.0f;
                var_y = 0.0f;
                cov_xy = 0.0f;
            }

            features[FTINDEX(scale, label, FT_PROPORTION)] = proportion;
            features[FTINDEX(scale, label, FT_MX)] = mean_x;
            features[FTINDEX(scale, label, FT_MY)] = mean_y;
            features[FTINDEX(scale, label, FT_CXX)] = var_x;
            features[FTINDEX(scale, label, FT_CYY)] = var_y;
            features[FTINDEX(scale, label, FT_CXY)] = cov_xy;
        }
    }
}

void CLabelFeature::SetImage(const cv::Mat &imgNew)
{
    imgLabel = imgNew;
    calcIntImages();
}
