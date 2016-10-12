#pragma once

#if defined(SIMPLE_HOUSE)
#include "SimpleHouse.cuh"
#elif defined(MENGER_SPONGE)
#include "MengerSponge.cuh"
#elif defined(_3D_TREE_12_2)
#include "3DTree_12_2.cuh"
#elif defined(_3D_TREE_4_3)
#include "3DTree_4_3.cuh"
#elif defined(_3D_TREE_8_3)
#include "3DTree_8_3.cuh"
#elif defined(MC_SPACESHIP)
#include "MC_Spaceship.cuh"
#elif defined(MC_SKYSCRAPERS)
#include "MC_Skyscrapers.cuh"
#elif defined(SUBURBAN_HOUSE)
#include "SuburbanHouse.cuh"
#elif defined(BALCONY)
#include "Balcony.cuh"
#elif defined(COMMERCIAL)
#include "Commercial.cuh"
#elif defined(IMPORT)
#include "Import.cuh"
#else
#error undefined scene
#endif
