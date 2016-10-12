#pragma once

namespace PGA
{
	// WARNING: You cannot add new enum without changing the float packing algorithm!!!
	enum OperandType
	{
		ORT_SCALAR = 0,
		ORT_RAND,
		ORT_SHAPE_ATTR,
		ORT_OP

	};

}
