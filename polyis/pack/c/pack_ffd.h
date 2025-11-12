#ifndef PACK_FFD_H
#define PACK_FFD_H

#include "utilities.h"

typedef enum PackMode {
    Easiest_Fit,  // Pack into collage with most empty space
    First_Fit,    // Pack into first collage that fits
    Best_Fit      // Pack into collage with least empty space that fits
} PackMode;

CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w, PackMode mode);

#endif // PACK_FFD_H
