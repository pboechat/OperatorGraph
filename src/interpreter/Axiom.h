#pragma once

#include <math/vector.h>

#include <pga/compiler/ShapeType.h>

#include "Constants.h"

struct Axiom
{
	PGA::Compiler::ShapeType shapeType;
	int entryIndex;
	math::float2 vertices[::Constants::MaxNumVertices];
	unsigned int numVertices;

};
