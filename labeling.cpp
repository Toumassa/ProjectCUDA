/*
	labeling.cpp

	Copyright 2012 Julien Mille (julien.mille@liris.cnrs.fr)

*/

#include "labeling.h"

#include <string.h>
#include <float.h>
#include <stdio.h>
#include <iostream>
#include <list>
#include <vector>

using namespace std;

/**********************************************************************
*                       CLabeling                           *
**********************************************************************/

CArray1D<CLabelClassProperties> CLabeling::vectClassesProperties;

CLabeling::CLabeling()
{
	// Draw parameters
	fDisplayOpacity = 0.5f;

	bDisplayRegionPixels = true;
	bDisplayBand = false;
	bDisplayBoundary = false;
}

bool CLabeling::InitClassesProperties()
{
    FILE *pFileConfig;
    int iIndexClassMax, iIndexClass, iLine; //, iChar;
    char strLine[500], *strId;

    pFileConfig = fopen("classes_config.txt", "rb");
    if (pFileConfig==NULL)
    {
        cout<<"ERROR: cannot open classes_config.txt"<<endl;
        return false;
    }
    else
        cout<<"Parsing file classes_config.txt..."<<endl;

    iLine = 1;
    iIndexClassMax = -1;
    while (fgets(strLine, 500, pFileConfig)!=NULL)
    {
        // iChar = 0;
        strId = strstr(strLine, "id=");
        if (strId!=NULL)
            sscanf(strId+3, "%d", &iIndexClass);

        if (iIndexClass>iIndexClassMax)
            iIndexClassMax = iIndexClass;
        iLine++;
    }
    fclose(pFileConfig);

    if (iIndexClassMax<=0)
    {
        cout<<"ERROR: invalid maximum class index"<<endl;
        return false;
    }

    vectClassesProperties.Init(iIndexClassMax+1);

    vectClassesProperties[0].strClassName="Unlabeled";
    vectClassesProperties[0].rgbClassColor.Set(0,0,0);

    for (iIndexClass=1; iIndexClass<=iIndexClassMax; iIndexClass++)
    {
        vectClassesProperties[iIndexClass].strClassName="Unknown";
        vectClassesProperties[iIndexClass].rgbClassColor.Set(0,0,0);
    }

    pFileConfig = fopen("classes_config.txt", "rb");
    if (pFileConfig==NULL)
    {
        cout<<"ERROR: cannot reopen classes_config.txt"<<endl;
        return false;
    }

    iLine = 1;
    while (fgets(strLine, 500, pFileConfig)!=NULL)
    {
        // iChar = 0;
        strId = strstr(strLine, "id=");
        if (strId!=NULL)
        {
            sscanf(strId+3, "%d", &iIndexClass);
            if (iIndexClass>0 && iIndexClass<=iIndexClassMax)
            {
                CLabelClassProperties *pProp = &(vectClassesProperties[iIndexClass]);
                char *strNameLoc;
                strNameLoc = strstr(strLine, "name=");
                if (strNameLoc!=NULL)
                {
                    if (strNameLoc[5]=='\"')
                    {
                        char *strEndQuote;

                        strNameLoc+=6;
                        strEndQuote = strstr(strNameLoc,"\"");
                        if (strEndQuote!=NULL)
                        {
                            char strName[500];

                            strEndQuote[0] = 0;
                            strcpy(strName, strNameLoc);
                            pProp->strClassName = strName;
                            strEndQuote[0] = '\"';

                            char *strRgbLoc;
                            strRgbLoc = strstr(strLine, "rgb=");
                            if (strRgbLoc!=NULL)
                            {
                                int iRed, iGreen, iBlue;
                                sscanf(strRgbLoc+5, "%d,%d,%d", &iRed, &iGreen, &iBlue);
                                if (iRed>=0 && iRed<=255)
                                    pProp->rgbClassColor.byteRed = (unsigned char)iRed;
                                else
                                    cout<<"WARNING: line "<<iLine<<": invalid red value in 'rgb' field"<<endl;

                                if (iGreen>=0 && iGreen<=255)
                                    pProp->rgbClassColor.byteGreen = (unsigned char)iGreen;
                                else
                                    cout<<"WARNING: line "<<iLine<<": invalid green value in 'rgb' field"<<endl;

                                if (iBlue>=0 && iBlue<=255)
                                    pProp->rgbClassColor.byteBlue = (unsigned char)iBlue;
                                else
                                    cout<<"WARNING: line "<<iLine<<": invalid blue value in 'rgb' field"<<endl;
                                cout<<"Class id="<<iIndexClass<<" name="<<pProp->strClassName<<" rgb=("<<iRed<<","<<iGreen<<","<<iBlue<<")"<<endl;
                            }
                            else
                                cout<<"WARNING: line "<<iLine<<": cannot find 'rgb' field"<<endl;
                        }
                        else
                            cout<<"WARNING: line "<<iLine<<": cannot find ending '\"' for 'name' field"<<endl;
                    }
                    else
                        cout<<"WARNING: line "<<iLine<<": expected '\"' after 'name='"<<endl;
                }
                else
                    cout<<"WARNING: line "<<iLine<<": cannot find 'name' field"<<endl;
            }
            else
                cout<<"WARNING: line "<<iLine<<": invalid id="<<iIndexClass<<endl;
        }
        else {
            cout<<"WARNING: line "<<iLine<<": cannot find 'id' field"<<endl;
        }
        iLine++;
    }
    fclose(pFileConfig);

    return true;
}

void CLabeling::InitCircle(const CCouple<float> &pfCenter, float fRadius, unsigned int uiLabel)
{
	CCouple<int> piCenter, p, pmin, pmax;
	int iRadius, y2;

    if (uiLabel==0)
    {
        cerr<<"ERROR in CLabeling::InitCircle(...): not for background region"<<endl;
		return;
    }

	EmptyRegion(uiLabel);

	piCenter = (CCouple<int>)pfCenter;
	iRadius = (int)fRadius;

	if (piCenter.y<iRadius)
		pmin.y = -piCenter.y;
	else pmin.y = -iRadius;

	if (piCenter.y+iRadius>=iHeight)
		pmax.y = iHeight-piCenter.y-1;
	else pmax.y = iRadius;

	// Fill circle
	for (p.y=pmin.y; p.y<=pmax.y; p.y++)
	{
		y2 = (int)sqrt((float)(iRadius*iRadius-p.y*p.y));
		if (piCenter.x<y2)
			pmin.x = -piCenter.x;
		else pmin.x = -y2;

		if (piCenter.x+y2>=iWidth)
			pmax.x = iWidth-piCenter.x-1;
		else pmax.x = y2;

		for (p.x=pmin.x; p.x<=pmax.x; p.x++)
			if (Element(piCenter+p).HasLabel(0)==true)
                SetPixelLabel(piCenter + p, uiLabel);
	}
}

void CLabeling::InitEllipse(const CCouple<float> &pfCenter, float fRadiusX, float fRadiusY, unsigned int uiLabel)
{
	CCouple<int> piCenter, p, pmin, pmax;
	int iRadiusX, iRadiusY, y2;

    if (uiLabel==0)
    {
        cerr<<"ERROR in CLabeling::InitCircle(...): not for background region"<<endl;
		return;
    }

	EmptyRegion(uiLabel);

	piCenter = (CCouple<int>)pfCenter;
	iRadiusX = (int)fRadiusX;
	iRadiusY = (int)fRadiusY;

	if (piCenter.y<iRadiusY)
		pmin.y = -piCenter.y;
	else pmin.y = -iRadiusY;

	if (piCenter.y+iRadiusY>=iHeight)
		pmax.y = iHeight-piCenter.y-1;
	else pmax.y = iRadiusY;

	// Fill circle
	for (p.y=pmin.y; p.y<=pmax.y; p.y++)
	{
		y2 = (int)sqrt((float)((iRadiusY*iRadiusY-p.y*p.y)*iRadiusX)/(float)iRadiusY);

		if (piCenter.x<y2)
			pmin.x = -piCenter.x;
		else pmin.x = -y2;

		if (piCenter.x+y2>=iWidth)
			pmax.x = iWidth-piCenter.x-1;
		else pmax.x = y2;

		for (p.x=pmin.x; p.x<=pmax.x; p.x++)
            if (Element(piCenter+p).HasLabel(0)==true)
                SetPixelLabel(piCenter + p, uiLabel);
	}
}

void CLabeling::InitFromBinaryImage(const cv::Mat &imgInit, unsigned int uiLabel)
{
	CCouple<int> p;
	CImage2DPixel *pBits;
	int iOffsetEndRow;

    if (uiLabel==0)
    {
        cerr<<"ERROR in CLabeling::InitCircle(...): not for background region"<<endl;
		return;
    }

	EmptyRegion(uiLabel);

	if (imgInit.GetBitsPerPixel()!=8)
	{
		cerr<<"ERROR in CLabeling::InitFromBinaryImage(...): input image is not 8-bit."<<endl;
		return;
	}

	if (imgInit.GetWidth()!=iWidth || imgInit.GetHeight()!=iHeight)
	{
		cerr<<"ERROR in CLabeling::InitFromBinaryImage(...): input image size does not match with region size."<<endl;
		return;
	}

	iOffsetEndRow = imgInit.GetBytesPerLine() - imgInit.GetWidth();
	pBits = imgInit.GetBits();

	for (p.y=0; p.y<imgInit.GetHeight(); p.y++)
	{
		for (p.x=0; p.x<imgInit.GetWidth(); p.x++)
		{
			if (*pBits>127)
				SetPixelLabel(p, uiLabel);
			pBits++;
		}
		pBits += iOffsetEndRow;
	}
}

void CLabeling::InitFromBinaryImageWithOffset(const cv::Mat &imgInit, const CCouple<int> &piOffset, unsigned int uiLabel)
{
	CCouple<int> p, piMin, piMax;
	CImage2DPixel *pBits;
	int iOffsetEndRow;

    if (uiLabel==0)
    {
        cerr<<"ERROR in CLabeling::InitCircle(...): not for background region"<<endl;
		return;
    }

	EmptyRegion(uiLabel);

	if (imgInit.GetBitsPerPixel()!=8)
	{
		cerr<<"ERROR in CLabeling::InitFromBinaryImageWithOffset(...): input image is not 8-bit."<<endl;
		return;
	}

	iOffsetEndRow = imgInit.GetBytesPerLine() - imgInit.GetWidth();
	pBits = imgInit.GetBits();
	piMin = -piOffset;
	piMax = CCouple<int>(iWidth-1, iHeight-1)-piOffset;

	for (p.y=0; p.y<imgInit.GetHeight(); p.y++)
	{
		for (p.x=0; p.x<imgInit.GetWidth(); p.x++)
		{
			if (*pBits==255 && p.IsInRange(piMin, piMax))
				SetPixelLabel(p + piOffset, uiLabel);
			pBits++;
		}
		pBits += iOffsetEndRow;
	}
}

void CLabeling::CopyToBinaryImage(cv::Mat &imgOutput, unsigned int uiLabel) const
{
	CCouple<int> p;
	CImage2DPixel *pBits;
	int iOffsetEndRow;

	if (imgOutput.GetBitsPerPixel()!=8 || imgOutput.GetSize()!=GetSize())
	{
		if (imgOutput.Create(iWidth, iHeight, 8)==false)
			return;
	}

	iOffsetEndRow = imgOutput.GetBytesPerLine() - iWidth;
	pBits = imgOutput.GetBits();

	for (p.y=0; p.y<iHeight; p.y++)
	{
		for (p.x=0; p.x<iWidth; p.x++)
		{
			if (Element(p).HasLabel(uiLabel))
				*pBits = 255;
			else
				*pBits = 0;

			pBits++;
		}
		pBits += iOffsetEndRow;
	}
}

/*
void CLabeling::FillHoles(unsigned int uiLabel)
{
	cv::Mat imgFill;
	CLabelPixel *pPixel;
	CImage2DPixel *pBits;
	CCouple<int> piCurrent, piStart, piEnd;
	int iImageWidth, iImageHeight, iOffsetEndRow, iOffsetEndRowImage;

	if (vectRegionsProperties[uiLabel]->iNbPixels==0)
		return;

	piStart = coupleMax(vectRegionsProperties[uiLabel]->puiLabelMin - CCouple<int>(1), CCouple<int>(0));
	piEnd   = coupleMin(vectRegionsProperties[uiLabel]->puiLabelMax + CCouple<int>(1), CCouple<int>(iWidth-1, iHeight-1));
	iImageWidth = piEnd.x - piStart.x + 1;
	iImageHeight = piEnd.y - piStart.y + 1;

	imgFill.Create(iImageWidth, iImageHeight, 8);

	iOffsetEndRow = iWidth - iImageWidth;
	iOffsetEndRowImage = imgFill.GetBytesPerLine() - iImageWidth;

	pPixel = this->pElements + GetOffset(piStart);
	pBits = imgFill.GetBits();
	for (piCurrent.y=piStart.y; piCurrent.y<=piEnd.y; piCurrent.y++)
	{
		for (piCurrent.x=piStart.x; piCurrent.x<=piEnd.x; piCurrent.x++)
		{
			if (pPixel->IsInRegion(uiLabel)==true)
				*pBits = 255;
			else
				*pBits = 0;
			pPixel++;
			pBits++;
		}
		pPixel += iOffsetEndRow;
		pBits += iOffsetEndRowImage;
	}

	imgFill.RegionFill(CCouple<int>(0), 127);

	pBits = imgFill.GetBits();
	for (piCurrent.y=piStart.y; piCurrent.y<=piEnd.y; piCurrent.y++)
	{
		for (piCurrent.x=piStart.x; piCurrent.x<=piEnd.x; piCurrent.x++)
		{
			if (*pBits==0)
				SetPixelLabel(piCurrent, uiLabel);
			pBits++;
		}
		pBits += iOffsetEndRowImage;
	}

	UpdateAfterEvolution();
}*/

void CLabeling::AddCircle(const CCouple<int> &piCenter, int iRadius, unsigned int uiLabel, bool bReplaceAll)
{
	CCouple<int> p, pmin, pmax;
	int y2;

    if (uiLabel==0)
    {
        cerr<<"WARNING in CLabeling::AddCircle(...): cannot add pixels in background region"<<endl;
		return;
    }

	if (piCenter.y<iRadius)
		pmin.y = -piCenter.y;
	else pmin.y = -iRadius;

	if (piCenter.y+iRadius>=iHeight)
		pmax.y = iHeight-piCenter.y-1;
	else pmax.y = iRadius;

	// Fill circle
	for (p.y=pmin.y; p.y<=pmax.y; p.y++)
	{
		y2 = (int)sqrt((float)(iRadius*iRadius-p.y*p.y));
		if (piCenter.x<y2)
			pmin.x = -piCenter.x;
		else pmin.x = -y2;

		if (piCenter.x+y2>=iWidth)
			pmax.x = iWidth-piCenter.x-1;
		else pmax.x = y2;

		for (p.x=pmin.x; p.x<=pmax.x; p.x++)
            if ((bReplaceAll==false && Element(piCenter+p).HasLabel(0)) || bReplaceAll==true)
                SetPixelLabel(piCenter + p, uiLabel);
	}
}

void CLabeling::RemoveCircle(const CCouple<int> &piCenter, int iRadius, unsigned int uiLabel, bool bRemoveAll)
{
	CCouple<int> p, pmin, pmax;
	int y2;

    if (uiLabel==0)
    {
        cerr<<"WARNING in CLabeling::removeCircle(...): cannot remove pixels from background region"<<endl;
		return;
    }

	if (piCenter.y<iRadius)
		pmin.y = -piCenter.y;
	else pmin.y = -iRadius;

	if (piCenter.y+iRadius>=iHeight)
		pmax.y = iHeight-piCenter.y-1;
	else pmax.y = iRadius;

	// Fill circle
	for (p.y=pmin.y; p.y<=pmax.y; p.y++)
	{
		y2 = (int)sqrt((float)(iRadius*iRadius-p.y*p.y));
		if (piCenter.x<y2)
			pmin.x = -piCenter.x;
		else pmin.x = -y2;

		if (piCenter.x+y2>=iWidth)
			pmax.x = iWidth-piCenter.x-1;
		else pmax.x = y2;

		for (p.x=pmin.x; p.x<=pmax.x; p.x++)
            if ((bRemoveAll==false && Element(piCenter + p).HasLabel(uiLabel)) || bRemoveAll==true)
                SetPixelLabel(piCenter + p, 0);
	}
}

void CLabeling::EmptyRegion(unsigned int uiLabel)
{
	CLabelPixel *pPixel;
	CCouple<int> piCurrent;

	// Set all pixels out of region
	pPixel = pElements;
	for (piCurrent.y=0; piCurrent.y<iHeight; piCurrent.y++)
	{
	    for (piCurrent.x=0; piCurrent.x<iWidth; piCurrent.x++)
	    {
	        if (pPixel->HasLabel(uiLabel))
                SetPixelLabel(piCurrent, 0);
            pPixel++;
	    }
	}
}

void CLabeling::Empty()
{
	CLabelPixel *pPixel;
	int i;

	// Set all pixels out of region
	pPixel = pElements;
	for (i=0; i<iSize; i++)
	{
		pPixel->SetLabel(0);
		pPixel++;
	}
}

bool CLabeling::IsEmpty()
{
	CLabelPixel *pPixel;
	// unsigned int uiLabel;
	int i;

    if (iSize==0)
        return true;

    // Check all pixels
	pPixel = pElements;
	for (i=0; i<iSize && pPixel->GetLabel()==0; i++)
		pPixel++;

	if (i==iSize)
        return true;
    else
        return false;
}

bool CLabeling::InitFromOtherRegion(const CLabeling &regionOther)
{
	// CLabelPixel *pPixel;
	// const CLabelPixel *pPixelOther;
	// unsigned int uiLabel;
	// int i;

    if (regionOther.iWidth!=iWidth || regionOther.iHeight!=iHeight)
    {
        cout<<"ERROR in CLabeling::InitFromOtherRegion(...): regions have different sizes"<<endl;
        return false;
    }

    // Check all pixels
	/*
	pPixel = pElements;
	pPixelOther = regionOther.pElements;
	for (i=0; i<iSize; i++)
    {
        *pPixel = *pPixelOther;
        pPixel++;
        pPixelOther++;
    }*/
    memcpy(pElements, regionOther.pElements, iSize*sizeof(CLabelPixel));
    return true;
}

void CLabeling::MakeLabelImage(cv::Mat &imgOutput) const
{
    CCouple<int> p;
    CLabelPixel *pRegionPixel;
	CImage2DPixel *pBits;
	int iOffsetEndRow;

	if (imgOutput.type()!=CV_8UC1 || imgOutput.GetSize()!=GetSize())
	{
		if (imgOutput.Create(iWidth, iHeight, 8)==false)
			return;
	}

	iOffsetEndRow = imgOutput.GetBytesPerLine() - iWidth;
	pBits = imgOutput.GetBits();
    pRegionPixel = pElements;

	for (p.y=0; p.y<iHeight; p.y++)
	{
		for (p.x=0; p.x<iWidth; p.x++)
		{
			*pBits = (unsigned char)(pRegionPixel->GetLabel());
            pRegionPixel++;
			pBits++;
		}
		pBits += iOffsetEndRow;
	}
}

void CLabeling::MakeLabelImageRGB(cv::Mat &imgOutput) const
{
    CCouple<int> p;
    CLabelPixel *pRegionPixel;
	CImage2DPixel *pBits;
	int iOffsetEndRow;

	if (imgOutput.GetBitsPerPixel()!=24 || imgOutput.GetSize()!=GetSize())
	{
		if (imgOutput.Create(iWidth, iHeight, 24)==false)
			return;
	}

	iOffsetEndRow = imgOutput.GetBytesPerLine() - 3*iWidth;
	pBits = imgOutput.GetBits();
    pRegionPixel = pElements;

	for (p.y=0; p.y<iHeight; p.y++)
	{
		for (p.x=0; p.x<iWidth; p.x++)
		{
			*((CImage2DByteRGBPixel *)pBits) = vectClassesProperties[pRegionPixel->GetLabel()].rgbClassColor;
            pRegionPixel++;
			pBits+=3;
		}
		pBits += iOffsetEndRow;
	}
}

void CLabeling::InitFromLabelImage(const cv::Mat &imgLabel)
{
	CCouple<int> p;
	const CImage2DPixel *pBits;
	int iOffsetEndRow;

    Empty();

	if (imgLabel.GetBitsPerPixel()!=8)
	{
		cerr<<"ERROR in CLabeling::InitFromLabelImage(...): input image is not 8-bit."<<endl;
		return;
	}

	if (imgLabel.GetWidth()!=iWidth || imgLabel.GetHeight()!=iHeight)
	{
		cerr<<"ERROR in CLabeling::InitFromLabelImage(...): label image size does not match region size."<<endl;
		return;
	}

	iOffsetEndRow = imgLabel.GetBytesPerLine() - imgLabel.GetWidth();
	pBits = imgLabel.GetBits();

	for (p.y=0; p.y<imgLabel.GetHeight(); p.y++)
	{
		for (p.x=0; p.x<imgLabel.GetWidth(); p.x++)
		{
			if (*pBits!=0)
            {
                if (*pBits>=vectClassesProperties.GetSize())
                {
                    cerr<<"ERROR in CLabeling::InitFromLabelImage(...): pixel ("<<p.x<<","<<p.y<<") has invalid label "<<(unsigned int)(*pBits)<<"."<<endl;
                    Empty();
                    return;
                }
                else
                    SetPixelLabel(p, *pBits);
            }
			pBits++;
		}
		pBits += iOffsetEndRow;
	}
}

void CLabeling::SetPixelLabel(const CCouple<int> &p, unsigned int uiLabelNew)
{
    CLabelPixel *pPixel;

    pPixel = pElements + GetOffset(p);
	pPixel->SetLabel(uiLabelNew);
}

void CLabeling::GetConnectedComponents(CArray2D<int> &arrayCCIndices,
                                           vector<pair<CCouple<int>, CCouple<int> > > &listCCBounds) const
{
	unsigned int iCurrentLabel;
	CLabelPixel *pPixel, *pPixelNew; //, *pPixelNeighbor;
	int *pCCIndex, *pCCIndexNew, *pCCIndexNeighbor;
	unsigned int i;
	int iNbCC;

	CCouple<int> p, seed;
	pair<CCouple<int>, CCouple<int> > iMinMaxBound;
	vector<CCouple<int> > nouveaux, nouveaux2;

	nouveaux.reserve(2*(iWidth+iHeight));
	nouveaux2.reserve(2*(iWidth+iHeight));

	arrayCCIndices.Init(iWidth, iHeight);
	arrayCCIndices.Fill(-1);

	listCCBounds.reserve(200);

	// borne_inf.init(0, 0);
	// borne_sup.init(iWidth-1, iHeight-1);

	// Voisinage 2D (4-connexe)
	/*
	CVecteur<CCouple<int> > voisinage2D(4);
	voisinage2D[0].init(-1,0);
	voisinage2D[1].init(1,0);
	voisinage2D[2].init(0,-1);
	voisinage2D[3].init(0,1);

	// Offset correspondant au voisinage 2D (4-connexe)
	CVecteur<int> offsets_voisinage2D(4);
	offsets_voisinage2D[0] = -1;
	offsets_voisinage2D[1] = 1;
	offsets_voisinage2D[2] = -iWidth;
	offsets_voisinage2D[3] = iWidth;

	CVecteur<int> offsets_voisinage_image2D(4);
	offsets_voisinage_image2D[0] = -1;
	offsets_voisinage_image2D[1] = 1;
	offsets_voisinage_image2D[2] = -iBytesPerLine;
	offsets_voisinage_image2D[3] = iBytesPerLine;
	*/

	iNbCC = 0;
	// iOffsetEndRow = iBytesPerLine - iWidth;
	pPixel = pElements;
	pCCIndex = arrayCCIndices.GetBuffer();

	for (seed.y=0;seed.y<iHeight;seed.y++)
	{
		for (seed.x=0;seed.x<iWidth;seed.x++)
		{
			if (pPixel->GetLabel()!=0 && *pCCIndex==-1)
			{
				// New region
				*pCCIndex = iNbCC;
				iCurrentLabel = pPixel->GetLabel();

				nouveaux.push_back(seed);
				iMinMaxBound.first = seed;
				iMinMaxBound.second = seed;

				while (nouveaux.size()!=0)
				{
					for (i=0;i<nouveaux.size();i++)
					{
						p = nouveaux[i];
						pPixelNew = pElements + iWidth*p.y + p.x;
						pCCIndexNew = arrayCCIndices.GetBuffer() + iWidth*p.y + p.x;

						if (p.x>0)
						{
							pCCIndexNeighbor = pCCIndexNew-1;
							if (*pCCIndexNeighbor==-1 && (pPixelNew-1)->GetLabel()==iCurrentLabel)
							{
								*pCCIndexNeighbor = iNbCC;
								nouveaux2.push_back(CCouple<int>(p.x-1, p.y));
								if (iMinMaxBound.first.x==p.x) // Decrease minimum x
									iMinMaxBound.first.x--;
							}
						}
						if (p.x<iWidth-1)
						{
							pCCIndexNeighbor = pCCIndexNew+1;
							if (*pCCIndexNeighbor==-1 && (pPixelNew+1)->GetLabel()==iCurrentLabel)
							{
								*pCCIndexNeighbor = iNbCC;
								nouveaux2.push_back(CCouple<int>(p.x+1, p.y));
								if (iMinMaxBound.second.x==p.x) // Increase maximum x
									iMinMaxBound.second.x++;
							}
						}
						if (p.y>0)
						{
							pCCIndexNeighbor = pCCIndexNew-iWidth;
							if (*pCCIndexNeighbor==-1 && (pPixelNew-iWidth)->GetLabel()==iCurrentLabel)
							{
								*pCCIndexNeighbor = iNbCC;
								nouveaux2.push_back(CCouple<int>(p.x, p.y-1));
								if (iMinMaxBound.first.y==p.y) // Decrease minimum y
									iMinMaxBound.first.y--;
							}
						}
						if (p.y<iHeight-1)
						{
							pCCIndexNeighbor = pCCIndexNew+iWidth;
							if (*pCCIndexNeighbor==-1 && (pPixelNew+iWidth)->GetLabel()==iCurrentLabel)
							{
								*pCCIndexNeighbor = iNbCC;
								nouveaux2.push_back(CCouple<int>(p.x, p.y+1));
								if (iMinMaxBound.second.y==p.y) // Increase maximum y
									iMinMaxBound.second.y++;
							}
						}
					}
					nouveaux = nouveaux2;
					nouveaux2.clear();
				}
				listCCBounds.push_back(iMinMaxBound);
				iNbCC++;
			}
			pPixel++;
			pCCIndex++;
		}
	}
}
