#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stddef.h>

// Structure definitions

typedef struct UShortArray {
    unsigned short *data;
    int size;
    int capacity;
} UShortArray;

typedef struct Polyomino {
    UShortArray mask;
    int offset_i;
    int offset_j;
} Polyomino;

typedef struct PolyominoArray {
    Polyomino *data;
    int size;
    int capacity;
} PolyominoArray;

// Function declarations

// Initialize an unsigned short array with initial capacity
int UShortArray_init(UShortArray *array, int initial_capacity);

// Push a value onto the array, expanding if necessary
int UShortArray_push(UShortArray *array, unsigned short value);

// Free the array's data (array itself is on stack memory)
void UShortArray_cleanup(UShortArray *array);

// Free the polyomino's mask array
void Polyomino_cleanup(Polyomino *polyomino);

// Initialize a polyomino array with initial capacity
int PolyominoArray_init(PolyominoArray *array, int initial_capacity);

// Push a polyomino onto the array, expanding if necessary
int PolyominoArray_push(PolyominoArray *array, Polyomino value);

// Free the polyomino array's data and all contained polyominos
void PolyominoArray_cleanup(PolyominoArray *array);

#endif // UTILITIES_H

