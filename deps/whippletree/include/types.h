
/*
* file created by    Markus Steinberger / steinberger ( at ) icg.tugraz.at
*
* modifications by
*/

#ifndef SOFTSHELL_TOOLS_TYPES_INCLUDED
#define SOFTSHELL_TOOLS_TYPES_INCLUDED

typedef unsigned int uint;
typedef unsigned short ushort;

namespace Softshell
{
  struct dim
  {
    union
    {
      struct
      {
        uint x, y, z;
      };
      uint d[3];
    };
    dim(uint _x, uint _y = 1, uint _z = 1) :
      x(_x), y(_y), z(_z)
    {
    }
  };
};
#endif //SOFTSHELL_TOOLS_TYPES_INCLUDED