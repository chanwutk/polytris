#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stddef.h>

// ============================================================================
// Structure Definitions
// ============================================================================

// Dynamic array of unsigned shorts
typedef struct UShortArray {
    unsigned short *data;
    int size;
    int capacity;
} UShortArray;

// Represents a 2D coordinate/point
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

// Polyomino with coordinate-based mask and offset information
typedef struct Polyomino {
    CoordinateArray mask;
    int offset_i;
    int offset_j;
} Polyomino;

// Dynamic array of polyominoes
typedef struct PolyominoArray {
    Polyomino *data;
    int size;
    int capacity;
} PolyominoArray;

// Represents a placement result
typedef struct Placement {
    int y;
    int x;
    int rotation;
} Placement;

// Represents a polyomino's position in a collage
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

// Dynamic array of unsigned char pointers (for collage occupied tiles pool)
typedef struct UCharPArray {
    unsigned char **data;  // Array of unsigned char pointers
    int size;              // Current number of elements
    int capacity;          // Allocated capacity
} UCharPArray;

// Dynamic array of integers (for empty space tracking)
typedef struct IntArray {
    int *data;      // Array of integers
    int size;       // Current number of elements
    int capacity;   // Allocated capacity
} IntArray;

// Structure to hold polyomino with frame index for sorting
typedef struct PolyominoWithFrame {
    CoordinateArray shape;
    int oy;
    int ox;
    int frame;
    int size;
} PolyominoWithFrame;

// Dynamic array of PolyominoWithFrame
typedef struct PolyominoWithFrameArray {
    PolyominoWithFrame *data;
    int size;
    int capacity;
} PolyominoWithFrameArray;

// ============================================================================
// Function Declarations - UShortArray
// ============================================================================

// Initialize an unsigned short array with initial capacity
int UShortArray_init(UShortArray *array, int initial_capacity);

// Push a value onto the array, expanding if necessary
int UShortArray_push(UShortArray *array, unsigned short value);

// Free the array's data (array itself is on stack memory)
void UShortArray_cleanup(UShortArray *array);

// ============================================================================
// Function Declarations - Polyomino and PolyominoArray
// ============================================================================

// Free the polyomino's mask array
void Polyomino_cleanup(Polyomino *polyomino);

// Initialize a polyomino array with initial capacity
int PolyominoArray_init(PolyominoArray *array, int initial_capacity);

// Push a polyomino onto the array, expanding if necessary
int PolyominoArray_push(PolyominoArray *array, Polyomino value);

// Free the polyomino array's data and all contained polyominos
void PolyominoArray_cleanup(PolyominoArray *array);

// ============================================================================
// Function Declarations - CoordinateArray
// ============================================================================

// Initialize a coordinate array
int CoordinateArray_init(CoordinateArray *arr, int initial_capacity);

// Push a coordinate to the array
int CoordinateArray_push(CoordinateArray *arr, Coordinate coord);

// Cleanup coordinate array
void CoordinateArray_cleanup(CoordinateArray *arr);

// ============================================================================
// Function Declarations - PolyominoPositionArray
// ============================================================================

// Initialize a PolyominoPositionArray
int PolyominoPositionArray_init(PolyominoPositionArray *arr, int initial_capacity);

// Push a PolyominoPosition to the array
int PolyominoPositionArray_push(PolyominoPositionArray *arr, PolyominoPosition pos);

// Cleanup PolyominoPositionArray
void PolyominoPositionArray_cleanup(PolyominoPositionArray *arr);

// ============================================================================
// Function Declarations - CollageArray
// ============================================================================

// Initialize a CollageArray
int CollageArray_init(CollageArray *list, int initial_capacity);

// Push a PolyominoPositionArray to the list
int CollageArray_push(CollageArray *list, PolyominoPositionArray arr);

// Cleanup CollageArray
void CollageArray_cleanup(CollageArray *list);

// ============================================================================
// Function Declarations - UCharPArray
// ============================================================================

// Initialize a UCharPArray
int UCharPArray_init(UCharPArray *arr, int initial_capacity);

// Push an unsigned char pointer to the array
int UCharPArray_push(UCharPArray *arr, unsigned char *value);

// Cleanup UCharPArray (two-level cleanup: frees stored pointers then array)
void UCharPArray_cleanup(UCharPArray *arr);

// ============================================================================
// Function Declarations - IntArray
// ============================================================================

// Initialize an IntArray
int IntArray_init(IntArray *arr, int initial_capacity);

// Push an integer to the array
int IntArray_push(IntArray *arr, int value);

// Cleanup IntArray
void IntArray_cleanup(IntArray *arr);

// ============================================================================
// Function Declarations - PolyominoWithFrameArray
// ============================================================================

// Initialize PolyominoWithFrameArray
int PolyominoWithFrameArray_init(PolyominoWithFrameArray *arr, int initial_capacity);

// Push to PolyominoWithFrameArray
int PolyominoWithFrameArray_push(PolyominoWithFrameArray *arr, PolyominoWithFrame item);

// Cleanup PolyominoWithFrameArray
void PolyominoWithFrameArray_cleanup(PolyominoWithFrameArray *arr);

#endif // UTILITIES_H

