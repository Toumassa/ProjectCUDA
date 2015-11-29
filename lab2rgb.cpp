#include <iostream>
#include <unistd.h>
#include <omp.h>
#include <sys/stat.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "label.h"

/***************************************************************************
 USAGE
 ***************************************************************************/

void usage (char *com) 
{
    std::cerr<< "usage: " << com << " <input-label-image> <output-rgb-image>\n";
    exit(1);
}


/***************************************************************************
 MAIN PROGRAM
 ***************************************************************************/

int main(int argc, char* argv[])
{
    char *inputfname=NULL, *outputfname=NULL;

    if (argc!=3)
        usage(*argv);
    else
    {
        inputfname = argv[1];
        outputfname = argv[2];
    }

    cv::Mat lab = cv::imread(inputfname, cv::IMREAD_GRAYSCALE);
    if (lab.data==NULL)
    {
        std::cerr <<"Failed to read label image " << inputfname << "\n";
        return false;
    }

    // Write RGB segmentation map
    cv::Mat rgb;
    convertLabelToRGB(lab, rgb);

    if (cv::imwrite(outputfname, rgb)==false)
    {
        std::cerr <<"Failed to write to "<<outputfname<<std::endl;
        exit(-1);
    } 
        
    std::cout << "Image converted.\n";

    return 0;
}
