/*
	multipleregion.h -> multiplelabel.h

	Copyright 2012 Julien Mille (julien.mille@liris.cnrs.fr)

	Header file of library implementing the multiple region extension of the binary
	level set implementation.
*/

#ifndef _MULTIPLEREGION_H
#define _MULTIPLEREGION_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

using namespace std;

// Class CLabelPixel
// Represents a single pixel and its associated region
class CLabelPixel
{
  // Members
  private:
	// Status byte
	// All relevant information about a binary region pixel holds in a single byte
	// The first three bits indicate domains which the pixel belongs to
	// Bit 0: region
	// Bit 1: boundary
	// Bit 2: narrow band
	// Whether the pixel belongs to the inner or outer boundary (similarly, the inner or outer narrow band)
	// is indicated by a combination of these bits
	// The evolution of the binary region involves several neighborhoods: the 4-connexity neighborhood
	// to determine boundary pixels or candidate pixels for growing, the smoothing neighborhood which size
	// depends on the standard deviation of the gaussian filter, the band neighborhood which size
	// depends on the band thickness, and the distance neighborhood for approximating the signed euclidean
	// distance to the boundary.
	// When the region boundary gets close to image borders, care should be
	// taken when scanning the different neighborhoods, so that resulting neighbors do not fall outside
	// the image domain. Instead of testing coordinates inside scanning loops, safe areas are used. A pixel
	// being in a safe area with respect to a given neighborhood can have its neighbors scanned without checking
	// if their coordinates are within the image domain.
	// Four bits in the status byte indicate whether the pixel belongs to safe areas or not.
	// Bit 3: grow-safe area
	// Bit 4: smoothing-safe area
	// Bit 5: band-safe area
	// Bit 6: distance-safe area
	// Testing these bits is faster than checking pixel coordinates in loops
	// The last bit indicates whether the pixel is marked for smoothing after evolution
	unsigned char status;
	unsigned char ucLabel;

  // Member functions
  public:
	inline CLabelPixel() {status = 0; ucLabel = 0;}

	inline void SetLabel(unsigned int uiLabel) {ucLabel = (unsigned char)uiLabel;}
	inline bool HasLabel(unsigned int uiLabel) const {return ucLabel==((unsigned char)uiLabel);}
    inline unsigned int GetLabel() const {return (unsigned int)ucLabel;}

    // Bit 0 is unused
	inline void AddIntoBoundary() {status |= 0x02;}        // Set bit 1
	inline void RemoveFromBoundary() {status &= 0xFD;}     // Erase bit 1
	inline bool IsInBoundary() const {return (status&0x02)!=0;}  // Test bit 1

	inline bool IsInInnerBoundary() const {return ((status&0x02)!=0 && ucLabel!=0);}  // Bit 1 should be set
	inline bool IsInOuterBoundary() const {return ((status&0x02)!=0 && ucLabel==0);}  // Bit 1 should be set

	inline void AddIntoNarrowBand() {status |= 0x04;}        // Set bit 2
	inline void RemoveFromNarrowBand() {status &= 0xFB;}     // Erase bit 2
	inline bool IsInNarrowBand() const {return (status&0x04)!=0;}  // Test bit 2

	inline bool IsInInnerNarrowBand() const {return ((status&0x04)!=0 && ucLabel!=0);}  // Bit 2 should be set
	inline bool IsInOuterNarrowBand() const {return ((status&0x04)!=0 && ucLabel==0);}  // Bit 2 should be set

	inline void AddIntoGrowSafeArea() {status |= 0x08;}        // Set bit 3
	inline void RemoveFromGrowSafeArea() {status &= 0xF7;}     // Erase bit 3
	inline bool IsInGrowSafeArea() const {return (status&0x08)!=0;}  // Test bit 3

	inline void AddIntoSmoothingSafeArea() {status |= 0x10;}        // Set bit 4
	inline void RemoveFromSmoothingSafeArea() {status &= 0xEF;}     // Erase bit 4
	inline bool IsInSmoothingSafeArea() const {return (status&0x10)!=0;}  // Test bit 4

	inline void AddIntoBandSafeArea() {status |= 0x20;}        // Set bit 5
	inline void RemoveFromBandSafeArea() {status &= 0xDF;}     // Erase bit 5
	inline bool IsInBandSafeArea() const {return (status&0x20)!=0;}  // Test bit 5

	inline void AddIntoDistanceSafeArea() {status |= 0x40;}        // Set bit 6
	inline void RemoveFromDistanceSafeArea() {status &= 0xBF;}     // Erase bit 6
	inline bool IsInDistanceSafeArea() const {return (status&0x40)!=0;}  // Test bit 6

	inline void MarkForSmoothing() {status |= 0x80;}               // Set bit 7
	inline void UnmarkForSmoothing() {status &= 0x7F;}             // Erase bit 7
	inline bool IsMarkedForSmoothing() const {return (status&0x80)!=0;}  // Test bit 7

	// Reset all data flags on current pixel, except region information
	// Clear all bits except bit 0
	inline void ClearStatusByte() {status &= 0x01;}
};

class CLabelClassProperties
{
  public:
    string strClassName;
    unsigned char rgbClassColor;
  public:
    CLabelClassProperties() {}
};

// Abstract base class CLabeling
// Contains pure virtual member functions and thus cannot be instantiated
class CLabeling : public CArray2D<CLabelPixel> //, public CDeformableModelBase
{
  // Static members
  public:
    static CArray1D<CLabelClassProperties> vectClassesProperties;

  // Members
  public:
	// Display parameters used in DrawInImageRGB()
	float fDisplayOpacity;     // Opacity for displaying
	bool bDisplayRegionPixels;  // Display region pixels with variable opacity
	bool bDisplayBand;          // Display narrow band ?
	bool bDisplayBoundary;      // Display boundary pixels ?

  // Static member functions
  public:
    static bool InitClassesProperties();
	// static void InitNeighborhood();

  // Member functions
  protected:
	// Add a single pixel into a region
	virtual void SetPixelLabel(const CCouple<int> &, unsigned int);

	// Check if the pixel at the given address is on the inner boundary
	// (belongs to the region and has at least neighbor which does not)
	inline bool IsInnerBoundaryPixel(CLabelPixel *pPixel) const {return pPixel->IsInInnerBoundary();}

	// Check if the pixel at the given address is on the outer boundary
	// (does not belong to the region and has at least a neighbor which does)
	inline bool IsOuterBoundaryPixel(CLabelPixel *pPixel) const {return pPixel->IsInOuterBoundary();}

  public:
	// Default constructor
	CLabeling();

	// Desctructor
	virtual ~CLabeling() {}

	// Initialize the region as a filled circle
	// Params: center, radius, region
	virtual void InitCircle(const CCouple<float> &, float, unsigned int);
    virtual void InitCircle(const CCouple<float> &pfCenter, float fRadius) {InitCircle(pfCenter, fRadius, 1);}

	// Initialize the region as a filled ellipse
	// Params: center, x-radius, y-radius, region
	virtual void InitEllipse(const CCouple<float> &, float, float, unsigned int);
    virtual void InitEllipse(const CCouple<float> &pfCenter, float fRadiusX, float fRadiusY) {InitEllipse(pfCenter, fRadiusX, fRadiusY, 1);}

	// Initialize the region from a binary image
	// Pixels with intensity different than 0 will be set inside the region
	// Params: input 8-bit image
	virtual void InitFromBinaryImage(const cv::Mat &, unsigned int);

	// Initialize the region from a binary image with offset translation
	// Pixels with intensity different than 0 will be set inside the region
	// Params: input 8-bit image
	virtual void InitFromBinaryImageWithOffset(const cv::Mat &, const CCouple<int> &, unsigned int);

	// Save region to image
	// Pixels inside region will be at 255
	// Params: output 8-bit image
	virtual void CopyToBinaryImage(cv::Mat &, unsigned int) const;

	// Fill holes
	// virtual void FillHoles(unsigned int);

    // Add a circle in region
    // Params : center, radius, target region index, boolean indicating if target region index should replace all current region indices
    virtual void AddCircle(const CCouple<int> &, int, unsigned int, bool bReplaceAll=false);

    // Remove a circle from region
    // Params : center, radius, target region index, boolean indicating if all current region indices should be removed
    // (if it is true, the target region index is ignored)
    virtual void RemoveCircle(const CCouple<int> &, int, unsigned int, bool bRemoveAll=false);

    // Empty a single region (not for background region)
    virtual void EmptyRegion(unsigned int);

    // Empty all regions
	virtual void Empty();

    virtual bool IsEmpty();

    virtual bool InitFromOtherRegion(const CLabeling &);

    virtual void MakeLabelImage(cv::Mat &) const;
    virtual void MakeLabelImageRGB(cv::Mat &) const;

    virtual void InitFromLabelImage(const cv::Mat &);

    void GetConnectedComponents(CArray2D<int> &, vector<pair<CCouple<int>, CCouple<int> > > &) const;
};

#endif
