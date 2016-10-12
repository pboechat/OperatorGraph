#pragma once

#include <pga/core/GlobalConstants.h>
#include <pga/core/DynamicPolygon.cuh>
#include <pga/core/DynamicRightPrism.cuh>

//////////////////////////////////////////////////////////////////////////
typedef PGA::Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, true> DCPoly;
typedef PGA::Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, false> DPoly;
typedef PGA::Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, true> DCRPrism;
typedef PGA::Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, false> DRPrism;

//////////////////////////////////////////////////////////////////////////
// Test control methods
std::string getTestName();
std::string getSceneName();
std::string getConfigurationString();
unsigned int getNumAxioms();
void incrementNumElements();
void decrementNumElements();
void incrementGridSize();
void decrementGridSize();
void incrementAttribute();
void decrementAttribute();
unsigned int getNumElements();
void setNumElements(unsigned int numElements);
void setAttributeIndex(unsigned int attributeIndex);
void setAttributeValue(unsigned int value);
void maximizeNumElements();
bool isInstrumented();

//////////////////////////////////////////////////////////////////////////
// PGA control methods
void initializePGA();
double executePGA();
void destroyPGA();
void releasePGA();

