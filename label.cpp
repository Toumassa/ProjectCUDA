#include "label.h"

unsigned char arrayClassesColors[14][3] = {
{0,0,0},
{0,127,0},
{0,0,255},
{255,255,0},
{255,0,0},
{255,215,0},
{244,165,90},
{255,193,193},
{173,255,47},
{0,250,154},
{135,206,250},
{255,62,150},
{255,255,255},
{127, 127, 127}
};

void convertLabelToRGB(const cv::Mat &imgLabel, cv::Mat &imgLabelRGB)
{
    cv::Point pt;
    unsigned char *pLabelRGB;
    const unsigned char *pLabel;

    if (imgLabelRGB.type()!=CV_8UC3 || imgLabelRGB.size()!=imgLabel.size())
        imgLabelRGB.create(imgLabel.size(), CV_8UC3);

    for (pt.y=0; pt.y<imgLabel.rows; pt.y++)
    {
        pLabel = imgLabel.ptr(pt.y);
        pLabelRGB = imgLabelRGB.ptr(pt.y);

        for (pt.x=0; pt.x<imgLabel.cols; pt.x++)
        {
            pLabelRGB[2] = arrayClassesColors[(*pLabel)%12][0];
            pLabelRGB[1] = arrayClassesColors[(*pLabel)%12][1];
            pLabelRGB[0] = arrayClassesColors[(*pLabel)%12][2];
            pLabel++;
            pLabelRGB+=3;
        }
    }
}
