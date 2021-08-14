#pragma once

#include "Constants.h"

#include <math/vector.h>
#include <pga/compiler/ShapeType.h>

struct Axiom
{
	PGA::Compiler::ShapeType shapeType;
	int entryIndex;
	math::float2 vertices[::Constants::MaxNumVertices];
	unsigned int numVertices;

};
