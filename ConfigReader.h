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

#ifndef CONFIGREADER_H_
#define CONFIGREADER_H_

#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

namespace vision
{
enum FOREST_TYPE
{
  CLASS
};
enum ENERGY_TYPE
{
  ENTROPY, STRUCTLABEL
};
enum RUN_TYPE
{
  TRAIN, TEST, EVAL
};
enum TRAIN_TYPE
{
  STD, INTERLEAVED
};
enum SAMPLING_TYPE
{
  HOMOGENEOUS, STRIDE
};
enum FEATURES2NDSTAGE_TYPE
{
  HOT1, OURS
};

class ConfigReader
{
public:

  /* tree and forest specifics */
  FOREST_TYPE forestType;
  ENERGY_TYPE energyType;
  RUN_TYPE runType;
  TRAIN_TYPE trainType;
  SAMPLING_TYPE samplingType;
  FEATURES2NDSTAGE_TYPE features2ndStage;

  int numTrees, maxProbeOffset;

  /* file, folder and experiment specifics */
  string imageFolder, groundTruthFolder, featureFolder, feature2ndStageFolder;
  string treeFolder, outputFolder, tree2ndStageFolder, output2ndStageFolder;
  string listImages, listTestImageNumbers, listTrainingImageNumbersPrefix;
  vector<string> imageFilenames;
  // vector<string> allTestFilenames;
  double samplingParam, rescaleFactor;
  double lambda;
  int maxDepth, minNumSamples, numNodeTests, numLabels;
  int labelPatchWidth, labelPatchHeight, jointProbDim;
  bool useWeights;

  /////////////////////////////////////////////////////

  ConfigReader();

  bool readConfigFile(string cfgFile);

  vector<string> readFilenames(string filename);

  ~ConfigReader()
  {
  }

};

} /* namespace vision */

#endif /* CONFIGREADER_H_ */

