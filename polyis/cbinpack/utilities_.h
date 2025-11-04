#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stddef.h>

// Structure definitions

typedef struct IntStack {
    unsigned short *data;
    int top;
    int capacity;
} IntStack;

typedef struct Polyomino {
    IntStack mask;
    int offset_i;
    int offset_j;
} Polyomino;

typedef struct PolyominoStack {
    Polyomino *data;
    int top;
    int capacity;
} PolyominoStack;

// Function declarations

// Initialize an integer stack with initial capacity
int IntStack_init(IntStack *stack, int initial_capacity);

// Push a value onto the stack, expanding if necessary
int IntStack_push(IntStack *stack, unsigned short value);

// Free the stack's data array (stack itself is on stack memory)
void IntStack_cleanup(IntStack *stack);

// Free the polyomino's mask stack
void Polyomino_cleanup(Polyomino *polyomino);

// Initialize a polyomino stack with initial capacity
int PolyominoStack_init(PolyominoStack *stack, int initial_capacity);

// Push a polyomino onto the stack, expanding if necessary
int PolyominoStack_push(PolyominoStack *stack, Polyomino value);

// Free the polyomino stack's data array and all contained polyominos
void PolyominoStack_cleanup(PolyominoStack *stack);

#endif // UTILITIES_H

