// =========================================================================================
//    Structured Class-Label in Random Forests. This is a re-implementation of
//    the work we presented at ICCV'11 in Barcelona, Spain.
//
//    In case of using this code, please cite the following paper:
//    P. Kontschieder, S. Rota Bulò, H. Bischof and M. Pelillo.
//    Structured Class-Labels in Random Forests for Semantic Image Labelling. In (ICCV), 2011.
//
//    Implementation by Peter Kontschieder and Samuel Rota Bulò
//    October 2013
//
// =========================================================================================

#include "ConfigReader.h"

using namespace std;

bool iequals(const string& a, const string& b)
{
    unsigned int sz = a.size();
    if (b.size() != sz)
        return false;
    for (unsigned int i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

namespace vision
{
  ConfigReader::ConfigReader()
  {
  }

  bool ConfigReader::readConfigFile(string cfgFile)
  {
    bool testImgFolderProvided = false;
    ifstream in(cfgFile.c_str());

    if(in.is_open())
    {
      string line;
      string delimiters = " =";
      while(in.good())
      {
        getline(in, line);
        // check for 0-length and if its a comment
        if(line.length() == 0 || line.find_first_of("#") != string::npos)
          continue;

        // expect a '=' somewhere in the line
        size_t pos2 = line.find_first_of("=");
        if(pos2 == string::npos)
        {
          cout << "Invalid configuration line in config file" << endl;
          in.close();
          return false;
        }

        size_t sep_pos = pos2;
        vector<string> substrings(2);
        vector<string> substrings_lower(2);

        // trim head and tail of first part
        size_t pos1 = 0;
        while(line[pos1] == ' ' && pos1 < pos2) ++pos1;
        --pos2; // jump one char left, away from "="
        while(line[pos2] == ' ' && pos1 < pos2) --pos2;
        substrings[0] = string(line.substr(pos1, pos2-pos1+1));

        // make it lowercase
        // transform(substrings[0].begin(), substrings[0].end(), substrings[0].begin(), ::tolower);

        pos1 = sep_pos + 1;
        pos2 = line.length() - 1;
        while(line[pos1] == ' ' && pos1 < pos2) ++pos1;
        while(line[pos2] == ' ' && pos1 < pos2) --pos2;
        substrings[1] = string(line.substr(pos1, pos2-pos1+1));
        // transform(substrings[1].begin(), substrings[1].end(), substrings[1].begin(), ::tolower);

        // do not modify pos1 and pos2 after here!

        // walk through parameters
        // if(iequals(substrings[0],"foresttype") == 0)
        if(iequals(substrings[0],"foresttype") == true)
        {
          if(iequals(substrings[1],"classification") == true)
            forestType = CLASS;
          else
            cout << "Unknown ForestType value (" << substrings[1] << ")" << endl;
        }
        else if(iequals(substrings[0],"energytype") == true)
        {
          if(iequals(substrings[1],"entropy") == true)
            energyType = ENTROPY;
          else if(iequals(substrings[1],"structlabel") == true)
            energyType = STRUCTLABEL;
          else
            cout << "Unknown EnergyType value (" << substrings[1] << ")" << endl;
        }
        else if(iequals(substrings[0],"traintype") == true)
        {
          if(iequals(substrings[1],"standard") == true)
            trainType = STD;
          else if(iequals(substrings[1],"interleaved") == true)
            trainType= INTERLEAVED;
          else
            cout << "Unknown TrainType value (" << substrings[1] << ")" << endl;
        }
        else if(iequals(substrings[0],"samplingtype") == true)
        {
          if(iequals(substrings[1],"homogeneous") == true)
            samplingType = HOMOGENEOUS;
          else if(iequals(substrings[1],"stride") == true)
            samplingType = STRIDE;
          else
            cout << "Unknown SamplingType value (" << substrings[1] << ")" << endl;
        }
        else if(iequals(substrings[0],"samplingparameter") == true)
        {
          samplingParam = atof(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"imagefolder") == true)
        {
          imageFolder = substrings[1];
          // cout<<"Image folder = "<<imageFolder<<endl;
        }
        else if(iequals(substrings[0],"groundtruthfolder") == true)
        {
          groundTruthFolder = substrings[1];
          // cout<<"Ground truth folder = "<<groundTruthFolder<<endl;
        }
        else if(iequals(substrings[0],"featurefolder") == true)
        {
          featureFolder = substrings[1];
          // cout<<"Feature folder = "<<featureFolder<<endl;
        }
        else if(iequals(substrings[0],"feature2ndstagefolder") == true)
        {
          feature2ndStageFolder = substrings[1];
          // cout<<"2nd-stage feature folder = "<<featureFolder<<endl;
        }
        else if(iequals(substrings[0],"treefolder") == true)
        {
          treeFolder = substrings[1];
          // cout<<"Tree folder = "<<treeFolder<<endl;
        }
        else if(iequals(substrings[0],"outputfolder") == true)
        {
          outputFolder = substrings[1];
          // cout<<"Output folder = "<<outputFolder<<endl;
        }
        else if(iequals(substrings[0],"tree2ndstagefolder") == true)
        {
          tree2ndStageFolder = substrings[1];
          // cout<<"2nd-stage tree folder = "<<tree2ndStageFolder<<endl;
        }
        else if(iequals(substrings[0],"output2ndstagefolder") == true)
        {
          output2ndStageFolder = substrings[1];
          // cout<<"2nd-stage output folder = "<<output2ndStageFolder<<endl;
        }
        else if(iequals(substrings[0],"maxdepth") == true)
        {
          maxDepth = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"minnumsamples") == true)
        {
          minNumSamples = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"numnodetests") == true)
        {
          numNodeTests = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"listimages") == true)
        {
          listImages = substrings[1];
          imageFilenames = readFilenames(listImages);
        }
        else if(iequals(substrings[0],"listtestimagenumbers") == true)
        {
          listTestImageNumbers = substrings[1];
        }
        else if(iequals(substrings[0],"listtrainingimagenumbersprefix") == true)
        {
          listTrainingImageNumbersPrefix = substrings[1];
        }
        else if(iequals(substrings[0],"maxprobeoffset") == true)
        {
          maxProbeOffset = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"rescalefactor") == true)
        {
          rescaleFactor = atof(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"numlabels") == true)
        {
          numLabels = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"useweights") == true)
        {
          useWeights = (iequals(substrings[0],"true") == true) ? true : false;
        }
        else if(iequals(substrings[0],"lambda") == true)
        {
          lambda = atof(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"labelpatchwidth") == true)
        {
          labelPatchWidth = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"labelpatchheight") == true)
        {
          labelPatchHeight = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"jointprobdim") == true)
        {
          jointProbDim = atoi(substrings[1].c_str());
        }
        else if(iequals(substrings[0],"features2ndstage") == true)
        {
          if(iequals(substrings[1],"hot1") == true)
            features2ndStage = HOT1;
          else if(iequals(substrings[1],"ours") == true)
            features2ndStage = OURS;
          else
            cout << "Unknown 2nd-stage feature value (" << substrings[1] << ")" << endl;
        }
        /* Unknown parameter */
        else
        {
          cout << "Unknown parameter in config file: (" << substrings[0] << ")" << endl;
        }
      }
    }
    else
    {
      return false;
    }

    in.close();

    // if(!testImgFolderProvided)
    //  testImageFolder = imageFolder;

    return true;
  }

  vector<string> ConfigReader::readFilenames(string filename)
  {
    vector<string> fnames;

    ifstream in(filename.c_str());
    if(in.is_open())
    {
      string line;
      while(in.good())
      {
        getline(in, line);
        if(line.length() == 0)
          continue;

        size_t pos = line.find_last_of("\n");
        if(pos != string::npos)
          line.replace(pos, 1, "");
        fnames.push_back(line);
      }
      in.close();
    }
    else
    {
      cout << "File name could not be opened: (" << filename << ")" << endl;
    }
    return fnames;
  }
}
