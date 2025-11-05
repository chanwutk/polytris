#ifndef PACK_FFD_H
#define PACK_FFD_H

#include "utilities.h"

// Structure to hold a coordinate/point
typedef struct Coordinate {
    int y;
    int x;
} Coordinate;

// Dynamic array of coordinates
typedef struct CoordinateArray {
    Coordinate *data;
    int size;
    int capacity;
} CoordinateArray;

// Structure representing a polyomino's position in a collage
typedef struct PolyominoPosition {
    int oy;              // Original y-offset from video frame
    int ox;              // Original x-offset from video frame
    int py;              // Packed y-position in collage
    int px;              // Packed x-position in collage
    int rotation;        // Rotation applied (0-3)
    int frame;           // Frame index
    CoordinateArray shape;  // Shape as coordinate array
} PolyominoPosition;

// Dynamic array of PolyominoPosition
typedef struct PolyominoPositionArray {
    PolyominoPosition *data;
    int size;
    int capacity;
} PolyominoPositionArray;

// List of collages (each collage contains multiple polyomino positions)
typedef struct CollageArray {
    PolyominoPositionArray *data;
    int size;
    int capacity;
} CollageArray;

// Function declarations
void CoordinateArray_cleanup(CoordinateArray *arr);
void PolyominoPositionArray_cleanup(PolyominoPositionArray *arr);
void CollageArray_cleanup(CollageArray *list);

// Main packing function
CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w);

#endif // PACK_FFD_H
