#include <stdlib.h>
#include <stddef.h>
// #include "utilities.h"

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

// Function implementations

// Initialize an integer stack with initial capacity
int IntStack_init(IntStack *stack, int initial_capacity) {
    // if not stack:
    //     return -1
    
    // Allocate memory for the stack data array
    stack->data = (unsigned short*)malloc((size_t)initial_capacity * sizeof(unsigned short));
    // if not stack.data:
    //     return -1
    
    // Initialize stack fields
    stack->top = 0;
    stack->capacity = initial_capacity;
    return 0;
}

// Push a value onto the stack, expanding if necessary
int IntStack_push(IntStack *stack, unsigned short value) {
    int new_capacity;
    unsigned short *new_data;
    
    // if not stack:
    //     return -1
    
    // Check if we need to expand the capacity
    if (stack->top >= stack->capacity) {
        // Double the capacity
        new_capacity = stack->capacity * 2;
        // Reallocate memory with the new capacity
        new_data = (unsigned short*)realloc((void*)stack->data,
                                            (size_t)new_capacity * sizeof(unsigned short));
        // if not new_data:
        //     return -1  // Memory allocation failed
        
        // Update stack data pointer and capacity
        stack->data = new_data;
        stack->capacity = new_capacity;
    }
    
    // Push the value onto the stack
    stack->data[stack->top] = value;
    stack->top += 1;
    return 0;
}

// Free the stack's data array (stack itself is on stack memory)
void IntStack_cleanup(IntStack *stack) {
    if (stack) {
        if (stack->data) {
            // Free the allocated data array
            free((void*)(stack->data));
            stack->data = NULL;
        }
        // Reset stack fields
        stack->top = 0;
        stack->capacity = 0;
    }
}

// Free the polyomino's mask stack
void Polyomino_cleanup(Polyomino *polyomino) {
    if (polyomino) {
        // Clean up the embedded IntStack mask
        IntStack_cleanup(&(polyomino->mask));
    }
}

// Initialize a polyomino stack with initial capacity
int PolyominoStack_init(PolyominoStack *stack, int initial_capacity) {
    // if not stack:
    //     return -1
    
    // Allocate memory for the polyomino array
    stack->data = (Polyomino*)malloc((size_t)initial_capacity * sizeof(Polyomino));
    // if not stack.data:
    //     return -1
    
    // Initialize stack fields
    stack->top = 0;
    stack->capacity = initial_capacity;
    return 0;
}

// Push a polyomino onto the stack, expanding if necessary
int PolyominoStack_push(PolyominoStack *stack, Polyomino value) {
    int new_capacity;
    Polyomino *new_data;
    
    // if not stack:
    //     return -1
    
    // Check if we need to expand the capacity
    if (stack->top >= stack->capacity) {
        // Double the capacity
        new_capacity = stack->capacity * 2;
        // Reallocate memory with the new capacity
        new_data = (Polyomino*)realloc((void*)stack->data,
                                       (size_t)new_capacity * sizeof(Polyomino));
        // if not new_data:
        //     return -1  // Memory allocation failed
        
        // Update stack data pointer and capacity
        stack->data = new_data;
        stack->capacity = new_capacity;
    }
    
    // Push the value onto the stack
    stack->data[stack->top] = value;
    stack->top += 1;
    return 0;
}

// Free the polyomino stack's data array and all contained polyominos
void PolyominoStack_cleanup(PolyominoStack *stack) {
    int i;
    if (stack) {
        if (stack->data) {
            // Clean up each polyomino in the stack
            for (i = 0; i < stack->top; i++) {
                Polyomino_cleanup(&(stack->data[i]));
            }
            // Free the allocated data array
            free((void*)stack->data);
            stack->data = NULL;
        }
        // Reset stack fields
        stack->top = 0;
        stack->capacity = 0;
    }
}

