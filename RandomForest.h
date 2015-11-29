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

#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdint.h>

#include "Global.h"

using namespace std;

namespace vision
{

// =====================================================================================
//        Class:  TNode
//  Description:
// =====================================================================================
template<class SplitData, class Prediction>
class TNode
{
public:
  // ====================  LIFECYCLE     =======================================
  TNode(int start, int end) :
      left(NULL), right(NULL), start(start), end(end), depth(0)
  {
    static int iNode = 0;
    idx = iNode++;

    // cout<<endl<<"New node "<<idx<<" "<<hex<<this<<dec<<endl;
  }

  ~TNode()
  {
    delete left;
    delete right;
  }

  // ====================  ACCESSORS     =======================================
  bool isLeaf() const
  {
    return left == NULL;
  }
  int getStart() const
  {
    return start;
  }
  int getEnd() const
  {
    return end;
  }
  int getIdx() const
  {
	  return idx;
  }
  int getNSamples() const
  {
    return end - start;
  }

  int getDepth() const
  {
    return depth;
  }

  const SplitData &getSplitData() const
  {
    return splitData;
  }
  const Prediction &getPrediction() const
  {
    return prediction;
  }

  const TNode<SplitData, Prediction>* getLeft() const
  {
    return left;
  }

  const TNode<SplitData, Prediction>* getRight() const
  {
    return right;
  }

  TNode<SplitData, Prediction>* getLeft()
  {
    return left;
  }

  TNode<SplitData, Prediction>* getRight()
  {
    return right;
  }

  // ====================  MUTATORS      =======================================

  void setSplitData(SplitData splitData)
  {
    this->splitData = splitData;
  }
  void setPrediction(Prediction prediction)
  {
    this->prediction = prediction;
  }

  void setDepth(uint16_t depth)
  {
    this->depth = depth;
  }

  void setEnd(uint32_t end)
  {
    this->end = end;
  }

  void setStart(uint32_t start)
  {
    this->start = start;
  }

  void split(uint32_t start, uint32_t middle)
  {
    assert(start >= this->start && middle >= start && middle <= end);
    if (left == NULL)
    {
      left = new TNode<SplitData, Prediction>(start, middle);
      right = new TNode<SplitData, Prediction>(middle, end);
      left->setDepth(depth + 1);
      right->setDepth(depth + 1);
    }
    else
    {
      left->setStart(start);
      left->setEnd(middle);
      right->setStart(middle);
    }
  }

  // ====================  OPERATORS     =======================================

protected:
  // ====================  METHODS       =======================================

  // ====================  DATA MEMBERS  =======================================

private:
  // ====================  METHODS       =======================================

  // ====================  DATA MEMBERS  =======================================

public:
  TNode<SplitData, Prediction> *left, *right;
  SplitData splitData;
  Prediction prediction;
  uint32_t start, end;
  uint16_t depth;
  uint32_t idx;

};
// -----  end of class TNode  -----

template<class Sample, class Label>
struct LabelledSample
{
  Sample sample;
  Label label;
};

enum SplitResult
{
  SR_LEFT = 0, SR_RIGHT = 1, SR_INVALID = 2
};

typedef vector<SplitResult> SplitResultsVector;

// =====================================================================================
//        Class:  RandomTree
//  Description:
// =====================================================================================
template<class SplitData, class Sample, class Label, class Prediction, class ErrorData>
class RandomTree
{
public:
  typedef LabelledSample<Sample, Label> LSample;
  typedef vector<LSample> LSamplesVector;
  // ====================  LIFECYCLE     =======================================
  RandomTree() :
      root(NULL)
  {
  }

  virtual ~RandomTree()
  {
    delete root;
    root = NULL;
  }

  // ====================  ACCESSORS     =======================================

  void save(string filename, bool includeSamples = false) const
  {
    ofstream out(filename.c_str());
    if (out.is_open()==false)
    {
        cout<<"Failed to open "<<filename<<endl;
        return;
    }
    writeHeader(out);
    out << endl;
    write(root, out);
    out << includeSamples << " ";
    if (includeSamples)
       write(samples, out);
  }

  Prediction predict(Sample &sample) const
  {
    assert(root != NULL);
    TNode<SplitData, Prediction> *curNode = root;
    SplitResult sr = SR_LEFT;
    while (!curNode->isLeaf() && sr != SR_INVALID)
      switch (sr = split(curNode->getSplitData(), sample))
      {
      case SR_LEFT:
        curNode = curNode->getLeft();
        break;
      case SR_RIGHT:
        curNode = curNode->getRight();
        break;
      default:
        break;
      }

    return curNode->getPrediction();
  }


  // ====================  MUTATORS      =======================================

  void train(const LSamplesVector &trainingSamples, int nTrials, bool interleavedTraining = false)
  {
    samples.clear();
    samples.resize(trainingSamples.size());
    splitResults.resize(trainingSamples.size());
    copy(trainingSamples.begin(), trainingSamples.end(), samples.begin());
    root = new TNode<SplitData, Prediction>(0, (int)samples.size());
    ErrorData errorData;
    Prediction rootPrediction;
    initialize(root, errorData, rootPrediction);
    root->setPrediction(rootPrediction);

    vector<TNode<SplitData, Prediction> *> nodeList[2], *curNodeList;
    nodeList[0].reserve(samples.size());
    nodeList[1].reserve(samples.size());

    int nodeListPtr = 0;
    curNodeList = &nodeList[nodeListPtr];

    curNodeList->push_back(root);

    cout<<"nTrials = "<<nTrials<<endl;

    while (curNodeList->size() > 0)
    {
      if (interleavedTraining)
      {
        //double initialError = getError(errorData);
        for (int t = 0; t < nTrials; ++t)
        {
          for (size_t i = 0; i < curNodeList->size(); ++i)
          {
            TNode<SplitData, Prediction> *node = (*curNodeList)[i];
            tryImprovingSplit(errorData, node);
          }
        }
        /*
         double finalError = getError(errorData);
         cout << "l " << level << " errorDelta: " << (finalError - initialError);
         cout << endl;
         */
      }
      else
      {
//        double initialError = getError(errorData);
        for (size_t i = 0; i < curNodeList->size(); ++i)
        {
          for (int t = 0; t < nTrials; ++t)
          {
            TNode<SplitData, Prediction> *node = (*curNodeList)[i];
            // cout<<"Node "<<node->idx<<" try "<<t<<endl;
            tryImprovingSplit(errorData, node);
          }
        }
        /*
         double finalError = getError(errorData);
         cout << "l " << level << " errorDelta: " << (finalError - initialError);
         cout << endl;
         */
      }

      int nextList = (++nodeListPtr) % 2;
      nodeList[nextList].clear();
      for (size_t i = 0; i < curNodeList->size(); ++i)
      {
        TNode<SplitData, Prediction> *node = (*curNodeList)[i];
        if (!node->isLeaf())
        {
#ifdef _DEBUG
          cout << setprecision(4) << (float)(node->getLeft()->getEnd() - node->getStart())/(node->getEnd() - node->getStart()) << "(" <<
            (node->getLeft()->getEnd() - node->getLeft()->getStart()) << ") / " <<
            setprecision(4) << (float)(node->getRight()->getEnd() - node->getRight()->getStart())/(node->getEnd() - node->getStart()) << "(" <<
            node->getRight()->getEnd() - node->getRight()->getStart() << ")" << endl;
#endif
          nodeList[nextList].push_back(node->getLeft());
          nodeList[nextList].push_back(node->getRight());
        }
        else
        {
          if(updateLeafPrediction(node, cLeftPrediction))
            node->setPrediction(cLeftPrediction);
        }
      }

      nodeListPtr = nextList;
      curNodeList = &nodeList[nextList];

    }
  }

  void load(string filename)
  {
    ifstream in(filename.c_str());
    readHeader(in);
    root = new TNode<SplitData, Prediction>(0, 0);
    read(root, in);
    bool includeSamples;
    in >> includeSamples;
    if (includeSamples)
      read(this->samples, in);
  }

  // ====================  OPERATORS     =======================================

protected:
  // ====================  METHODS       =======================================

  //virtual SplitData generateSplit(const TNode<SplitData, Prediction> *node) const=0;

  virtual SplitResult split(const SplitData &splitData, Sample &sample) const =0;

  virtual bool split(const TNode<SplitData, Prediction> *node, SplitData &splitData,
      Prediction &leftPrediction, Prediction &rightPrediction) = 0;

  virtual void initialize(const TNode<SplitData, Prediction> *node, ErrorData &errorData,
      Prediction &prediction) const = 0;

  virtual void updateError(ErrorData &newError, const ErrorData &errorData,
      const TNode<SplitData, Prediction> *node, Prediction &newLeft,
      Prediction &newRight) const = 0;

  virtual double getError(const ErrorData &error) const = 0;

  // non-pure virtual function which allows to modify predictions after all node split trials are made
  virtual bool updateLeafPrediction(const TNode<SplitData, Prediction> *node, Prediction &newPrediction) const
  {
    return false;
  }

  const LSamplesVector &getLSamples() const
  {
    return samples;
  }

  LSamplesVector &getLSamples()
  {
    return samples;
  }

  SplitResultsVector &getSplitResults()
  {
    return splitResults;
  }

  const SplitResultsVector &getSplitResults() const
  {
    return splitResults;
  }

  TNode<SplitData,Prediction>* getRoot() const
  {
    return root;
  }

  virtual void writeHeader(ostream &out) const=0;
  virtual void readHeader(istream &in) =0;

  virtual void write(const Sample &sample, ostream &out) const =0;
  virtual void read(Sample &sample, istream &in) const =0;

  virtual void write(const Prediction &prediction, ostream &out) const=0;
  virtual void read(Prediction &prediction, istream &in) const=0;

  virtual void write(const Label &label, ostream &out) const=0;
  virtual void read(Label &label, istream &in) const=0;

  virtual void write(const SplitData &splitData, ostream &out) const=0;
  virtual void read(SplitData &splitData, istream &in) const=0;

  // ====================  DATA MEMBERS  =======================================

protected:
  // ====================  METHODS       =======================================

  bool tryImprovingSplit(ErrorData &errorData, TNode<SplitData, Prediction> *node)
  {
    bool improved = false;

    if (split(node, cSplitData, cLeftPrediction, cRightPrediction))
    {

      double initialError = getError(errorData);
      ErrorData newErrorData;
      updateError(newErrorData, errorData, node, cLeftPrediction, cRightPrediction); //do not move this afterwards
      double deltaError = getError(newErrorData) - initialError;
      if (node->isLeaf() || deltaError < 0)
      {
        int start, middle;

        doSplit(node, start, middle);
        node->setSplitData(cSplitData);
        node->split(start, middle);
        node->getLeft()->setPrediction(cLeftPrediction);
        node->getRight()->setPrediction(cRightPrediction);
        errorData = newErrorData;
        improved = true;
      }
    }

    return improved;
  }

  void doSplit(const TNode<SplitData, Prediction> *node, int &pInvalid, int &pLeft)
  {
    pLeft = node->getStart();
    pInvalid = node->getStart();

    int pRight = node->getEnd() - 1;
    while (pLeft <= pRight)
    {
      LSample s;
      switch (splitResults[pLeft])
      {
      case SR_RIGHT:
        s = samples[pRight];
        samples[pRight] = samples[pLeft];
        samples[pLeft] = s;
        splitResults[pLeft] = splitResults[pRight];
        splitResults[pRight] = SR_RIGHT; //not necessary
        --pRight;
        break;
      case SR_INVALID:
        s = samples[pInvalid];
        samples[pInvalid] = samples[pLeft];
        samples[pLeft] = s;

        splitResults[pLeft] = splitResults[pInvalid];
        splitResults[pInvalid] = SR_INVALID;

        ++pInvalid;
        ++pLeft;
        break;
      case SR_LEFT:
        ++pLeft;
        break;
      }
    }
  }

  virtual void write(const TNode<SplitData, Prediction> *node, ostream &out) const
  {
    out << (node->isLeaf() ? "L " : "N ");
    out << node->getStart() << " " << node->getEnd() << " " << node->getDepth() << " ";
    write(node->getSplitData(), out);
    out << " ";
    write(node->getPrediction(), out);
    out << endl;
    if (!node->isLeaf())
    {
      write(node->getLeft(), out);
      write(node->getRight(), out);
    }
  }

  virtual void read(TNode<SplitData, Prediction> *node, istream &in) const
  {
    char type;
    in >> type;
    if (type!='L' && type!='N')
    {
        cout<<"ERROR: Unknown node type: "<<type<<endl;
        exit(-1);
    }
    bool isLeaf = type == 'L';

    int start, end, depth;
    in >> start;
    in >> end;
    in >> depth;
    node->setStart(start);
    node->setEnd(end);
    node->setDepth(depth);

    SplitData splitData;
    read(splitData, in);
    node->setSplitData(splitData);

    Prediction prediction;
    read(prediction, in);
    node->setPrediction(prediction);

    if (!isLeaf)
    {
      node->split(node->getStart(), node->getStart());
      read(node->getLeft(), in);
      read(node->getRight(), in);
    }
  }

  void write(const LSamplesVector &lSamples, ostream &out) const
  {
    out << lSamples.size() << " ";
    for (int i = 0; i < lSamples.size(); ++i)
    {
      write(lSamples[i].sample, out);
      out << " ";
      write(lSamples[i].label, out);
      out << " ";
    }
  }

  void read(LSamplesVector &lSamples, istream &in) const
  {
    int nSamples;
    in >> nSamples;
    lSamples.resize(nSamples);
    for (int i = 0; i < nSamples; ++i)
    {
      read(lSamples[i].sample, in);
      read(lSamples[i].label, in);
    }
  }

// ====================  DATA MEMBERS  =======================================

  TNode<SplitData, Prediction> *root;
  LSamplesVector samples;
  SplitResultsVector splitResults;

  SplitData cSplitData;
  Prediction cLeftPrediction, cRightPrediction;
};
// -----  end of class RandomTree  -----

}
#endif
