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

// Function implementations

// Initialize an unsigned short array with initial capacity
int UShortArray_init(UShortArray *array, int initial_capacity) {
    // if not array:
    //     return -1

    // Allocate memory for the array data
    array->data = (unsigned short*)malloc((size_t)initial_capacity * sizeof(unsigned short));
    // if not array.data:
    //     return -1

    // Initialize array fields
    array->size = 0;
    array->capacity = initial_capacity;
    return 0;
}

// Push a value onto the array, expanding if necessary
int UShortArray_push(UShortArray *array, unsigned short value) {
    int new_capacity;
    unsigned short *new_data;

    // if not array:
    //     return -1

    // Check if we need to expand the capacity
    if (array->size >= array->capacity) {
        // Double the capacity
        new_capacity = array->capacity * 2;
        // Reallocate memory with the new capacity
        new_data = (unsigned short*)realloc((void*)array->data,
                                            (size_t)new_capacity * sizeof(unsigned short));
        // if not new_data:
        //     return -1  // Memory allocation failed

        // Update array data pointer and capacity
        array->data = new_data;
        array->capacity = new_capacity;
    }

    // Push the value onto the array
    array->data[array->size] = value;
    array->size += 1;
    return 0;
}

// Free the array's data (array itself is on stack memory)
void UShortArray_cleanup(UShortArray *array) {
    if (array) {
        if (array->data) {
            // Free the allocated data array
            free((void*)(array->data));
            array->data = NULL;
        }
        // Reset array fields
        array->size = 0;
        array->capacity = 0;
    }
}

// Free the polyomino's mask array
void Polyomino_cleanup(Polyomino *polyomino) {
    if (polyomino) {
        // Clean up the embedded UShortArray mask
        UShortArray_cleanup(&(polyomino->mask));
    }
}

// Initialize a polyomino array with initial capacity
int PolyominoArray_init(PolyominoArray *array, int initial_capacity) {
    // if not array:
    //     return -1

    // Allocate memory for the polyomino array
    array->data = (Polyomino*)malloc((size_t)initial_capacity * sizeof(Polyomino));
    // if not array.data:
    //     return -1

    // Initialize array fields
    array->size = 0;
    array->capacity = initial_capacity;
    return 0;
}

// Push a polyomino onto the array, expanding if necessary
int PolyominoArray_push(PolyominoArray *array, Polyomino value) {
    int new_capacity;
    Polyomino *new_data;

    // if not array:
    //     return -1

    // Check if we need to expand the capacity
    if (array->size >= array->capacity) {
        // Double the capacity
        new_capacity = array->capacity * 2;
        // Reallocate memory with the new capacity
        new_data = (Polyomino*)realloc((void*)array->data,
                                       (size_t)new_capacity * sizeof(Polyomino));
        // if not new_data:
        //     return -1  // Memory allocation failed

        // Update array data pointer and capacity
        array->data = new_data;
        array->capacity = new_capacity;
    }

    // Push the value onto the array
    array->data[array->size] = value;
    array->size += 1;
    return 0;
}

// Free the polyomino array's data and all contained polyominos
void PolyominoArray_cleanup(PolyominoArray *array) {
    int i;
    if (array) {
        if (array->data) {
            // Clean up each polyomino in the array
            for (i = 0; i < array->size; i++) {
                Polyomino_cleanup(&(array->data[i]));
            }
            // Free the allocated data array
            free((void*)array->data);
            array->data = NULL;
        }
        // Reset array fields
        array->size = 0;
        array->capacity = 0;
    }
}

