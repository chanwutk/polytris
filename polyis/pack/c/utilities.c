#include <stdlib.h>
#include <stddef.h>
#include "errors.h"

// ============================================================================
// Macros for Dynamic Array Implementation
// ============================================================================

// Macro to define _init function for simple arrays
#define DEFINE_ARRAY_INIT(TypeName, ElementType) \
int TypeName##_init(TypeName *arr, int initial_capacity) { \
    CHECK_NULL(arr, #TypeName " array pointer is NULL"); \
    ASSERT(initial_capacity > 0, "initial_capacity must be positive"); \
    arr->data = (ElementType*)malloc((size_t)initial_capacity * sizeof(ElementType)); \
    CHECK_ALLOC(arr->data, "failed to allocate " #TypeName " array data"); \
    arr->size = 0; \
    arr->capacity = initial_capacity; \
    return 0; \
}

// Macro to define _push function for simple arrays
#define DEFINE_ARRAY_PUSH(TypeName, ElementType) \
int TypeName##_push(TypeName *arr, ElementType value) { \
    CHECK_NULL(arr, #TypeName " array pointer is NULL"); \
    CHECK_NULL(arr->data, #TypeName " array data is NULL"); \
    if (arr->size >= arr->capacity) { \
        int new_capacity = arr->capacity * 2; \
        ElementType *new_data = (ElementType*)realloc(arr->data, \
                                (size_t)new_capacity * sizeof(ElementType)); \
        CHECK_ALLOC(new_data, "failed to reallocate " #TypeName " array data"); \
        arr->data = new_data; \
        arr->capacity = new_capacity; \
    } \
    arr->data[arr->size] = value; \
    arr->size += 1; \
    return 0; \
}

// Macro to define _cleanup function for simple arrays (no nested cleanup)
#define DEFINE_ARRAY_CLEANUP_SIMPLE(TypeName) \
void TypeName##_cleanup(TypeName *arr) { \
    if (arr && arr->data) { \
        free(arr->data); \
        arr->data = NULL; \
    } \
    arr->size = 0; \
    arr->capacity = 0; \
}

// Macro to define complete simple array (init + push + cleanup)
#define DEFINE_SIMPLE_ARRAY(TypeName, ElementType) \
DEFINE_ARRAY_INIT(TypeName, ElementType) \
DEFINE_ARRAY_PUSH(TypeName, ElementType) \
DEFINE_ARRAY_CLEANUP_SIMPLE(TypeName)


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
    // Check input pointer is valid
    CHECK_NULL(array, "array pointer is NULL");

    // Check initial capacity is positive
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");

    // Allocate memory for the array data
    array->data = (unsigned short*)malloc((size_t)initial_capacity * sizeof(unsigned short));

    // Check if allocation succeeded
    CHECK_ALLOC(array->data, "failed to allocate array data");

    // Initialize array fields
    array->size = 0;
    array->capacity = initial_capacity;
    return 0;
}

// Push a value onto the array, expanding if necessary
int UShortArray_push(UShortArray *array, unsigned short value) {
    int new_capacity;
    unsigned short *new_data;

    // Check input pointer is valid
    CHECK_NULL(array, "array pointer is NULL");
    CHECK_NULL(array->data, "array data is NULL");

    // Check if we need to expand the capacity
    if (array->size >= array->capacity) {
        // Double the capacity
        new_capacity = array->capacity * 2;

        // Reallocate memory with the new capacity
        new_data = (unsigned short*)realloc((void*)array->data,
                                            (size_t)new_capacity * sizeof(unsigned short));

        // Check if reallocation succeeded
        CHECK_ALLOC(new_data, "failed to reallocate array data");

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
    // Check input pointer is valid
    CHECK_NULL(array, "array pointer is NULL");

    // Check initial capacity is positive
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");

    // Allocate memory for the polyomino array
    array->data = (Polyomino*)malloc((size_t)initial_capacity * sizeof(Polyomino));

    // Check if allocation succeeded
    CHECK_ALLOC(array->data, "failed to allocate polyomino array data");

    // Initialize array fields
    array->size = 0;
    array->capacity = initial_capacity;
    return 0;
}

// Push a polyomino onto the array, expanding if necessary
int PolyominoArray_push(PolyominoArray *array, Polyomino value) {
    int new_capacity;
    Polyomino *new_data;

    // Check input pointer is valid
    CHECK_NULL(array, "array pointer is NULL");
    CHECK_NULL(array->data, "array data is NULL");

    // Check if we need to expand the capacity
    if (array->size >= array->capacity) {
        // Double the capacity
        new_capacity = array->capacity * 2;

        // Reallocate memory with the new capacity
        new_data = (Polyomino*)realloc((void*)array->data,
                                       (size_t)new_capacity * sizeof(Polyomino));

        // Check if reallocation succeeded
        CHECK_ALLOC(new_data, "failed to reallocate polyomino array data");

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


// ============================================================================
// Structure Definitions
// ============================================================================

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

// ============================================================================
// Memory Management Functions for Dynamic Arrays
// ============================================================================

// Initialize a coordinate array
int CoordinateArray_init(CoordinateArray *arr, int initial_capacity) {
    CHECK_NULL(arr, "CoordinateArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    arr->data = (Coordinate*)malloc((size_t)initial_capacity * sizeof(Coordinate));
    CHECK_ALLOC(arr->data, "failed to allocate CoordinateArray data");
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push a coordinate to the array
int CoordinateArray_push(CoordinateArray *arr, Coordinate coord) {
    CHECK_NULL(arr, "CoordinateArray pointer is NULL");
    CHECK_NULL(arr->data, "CoordinateArray data is NULL");
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        Coordinate *new_data = (Coordinate*)realloc(arr->data,
                                                     (size_t)new_capacity * sizeof(Coordinate));
        CHECK_ALLOC(new_data, "failed to reallocate CoordinateArray data");
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the coordinate
    arr->data[arr->size] = coord;
    arr->size += 1;
    return 0;
}

// Cleanup coordinate array
void CoordinateArray_cleanup(CoordinateArray *arr) {
    if (arr && arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// Initialize a PolyominoPositionArray
int PolyominoPositionArray_init(PolyominoPositionArray *arr, int initial_capacity) {
    CHECK_NULL(arr, "PolyominoPositionArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    arr->data = (PolyominoPosition*)malloc((size_t)initial_capacity * sizeof(PolyominoPosition));
    CHECK_ALLOC(arr->data, "failed to allocate PolyominoPositionArray data");
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push a PolyominoPosition to the array
int PolyominoPositionArray_push(PolyominoPositionArray *arr, PolyominoPosition pos) {
    CHECK_NULL(arr, "PolyominoPositionArray pointer is NULL");
    CHECK_NULL(arr->data, "PolyominoPositionArray data is NULL");
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        PolyominoPosition *new_data = (PolyominoPosition*)realloc(arr->data,
                                                                   (size_t)new_capacity * sizeof(PolyominoPosition));
        CHECK_ALLOC(new_data, "failed to reallocate PolyominoPositionArray data");
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the position
    arr->data[arr->size] = pos;
    arr->size += 1;
    return 0;
}

// Cleanup PolyominoPositionArray
void PolyominoPositionArray_cleanup(PolyominoPositionArray *arr) {
    if (arr && arr->data) {
        // Cleanup each polyomino position's shape
        for (int i = 0; i < arr->size; i++) {
            CoordinateArray_cleanup(&arr->data[i].shape);
        }
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// Initialize a CollageArray
int CollageArray_init(CollageArray *list, int initial_capacity) {
    CHECK_NULL(list, "CollageArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    list->data = (PolyominoPositionArray*)malloc((size_t)initial_capacity * sizeof(PolyominoPositionArray));
    CHECK_ALLOC(list->data, "failed to allocate CollageArray data");
    list->size = 0;
    list->capacity = initial_capacity;
    return 0;
}

// Push a PolyominoPositionArray to the list
int CollageArray_push(CollageArray *list, PolyominoPositionArray arr) {
    CHECK_NULL(list, "CollageArray pointer is NULL");
    CHECK_NULL(list->data, "CollageArray data is NULL");
    // Expand if necessary
    if (list->size >= list->capacity) {
        int new_capacity = list->capacity * 2;
        PolyominoPositionArray *new_data = (PolyominoPositionArray*)realloc(list->data,
                                                                             (size_t)new_capacity * sizeof(PolyominoPositionArray));
        CHECK_ALLOC(new_data, "failed to reallocate CollageArray data");
        list->data = new_data;
        list->capacity = new_capacity;
    }
    // Push the array
    list->data[list->size] = arr;
    list->size += 1;
    return 0;
}

// Cleanup CollageArray
void CollageArray_cleanup(CollageArray *list) {
    if (list && list->data) {
        // Cleanup each collage
        for (int i = 0; i < list->size; i++) {
            PolyominoPositionArray_cleanup(&list->data[i]);
        }
        free(list->data);
        list->data = NULL;
    }
    list->size = 0;
    list->capacity = 0;
}

// Initialize a UCharPArray
int UCharPArray_init(UCharPArray *arr, int initial_capacity) {
    CHECK_NULL(arr, "UCharPArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    // Allocate array of pointers
    arr->data = (unsigned char**)malloc((size_t)initial_capacity * sizeof(unsigned char*));
    CHECK_ALLOC(arr->data, "failed to allocate UCharPArray data");
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an unsigned char pointer to the array
int UCharPArray_push(UCharPArray *arr, unsigned char *value) {
    CHECK_NULL(arr, "UCharPArray pointer is NULL");
    CHECK_NULL(arr->data, "UCharPArray data is NULL");
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        unsigned char **new_data = (unsigned char**)realloc(arr->data,
                                                            (size_t)new_capacity * sizeof(unsigned char*));
        CHECK_ALLOC(new_data, "failed to reallocate UCharPArray data");
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the pointer
    arr->data[arr->size] = value;
    arr->size += 1;
    return 0;
}

// Cleanup UCharPArray (two-level cleanup: frees stored pointers then array)
void UCharPArray_cleanup(UCharPArray *arr) {
    if (arr && arr->data) {
        // Free each stored pointer
        for (int i = 0; i < arr->size; i++) {
            if (arr->data[i]) {
                free(arr->data[i]);
            }
        }
        // Free the array itself
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// ============================================================================
// IntArray: Dynamic array for storing integers (for empty space tracking)
// ============================================================================

typedef struct IntArray {
    int *data;      // Array of integers
    int size;       // Current number of elements
    int capacity;   // Allocated capacity
} IntArray;

// Initialize an IntArray
int IntArray_init(IntArray *arr, int initial_capacity) {
    CHECK_NULL(arr, "IntArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    arr->data = (int*)malloc((size_t)initial_capacity * sizeof(int));
    CHECK_ALLOC(arr->data, "failed to allocate IntArray data");
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an integer to the array
int IntArray_push(IntArray *arr, int value) {
    CHECK_NULL(arr, "IntArray pointer is NULL");
    CHECK_NULL(arr->data, "IntArray data is NULL");
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        int *new_data = (int*)realloc(arr->data, (size_t)new_capacity * sizeof(int));
        CHECK_ALLOC(new_data, "failed to reallocate IntArray data");
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the value
    arr->data[arr->size] = value;
    arr->size += 1;
    return 0;
}

// Cleanup IntArray
void IntArray_cleanup(IntArray *arr) {
    if (arr && arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

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

// Initialize PolyominoWithFrameArray
int PolyominoWithFrameArray_init(PolyominoWithFrameArray *arr, int initial_capacity) {
    CHECK_NULL(arr, "PolyominoWithFrameArray pointer is NULL");
    ASSERT(initial_capacity > 0, "initial_capacity must be positive");
    arr->data = (PolyominoWithFrame*)malloc((size_t)initial_capacity * sizeof(PolyominoWithFrame));
    CHECK_ALLOC(arr->data, "failed to allocate PolyominoWithFrameArray data");
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push to PolyominoWithFrameArray
int PolyominoWithFrameArray_push(PolyominoWithFrameArray *arr, PolyominoWithFrame item) {
    CHECK_NULL(arr, "PolyominoWithFrameArray pointer is NULL");
    CHECK_NULL(arr->data, "PolyominoWithFrameArray data is NULL");
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        PolyominoWithFrame *new_data = (PolyominoWithFrame*)realloc(arr->data,
                                                                     (size_t)new_capacity * sizeof(PolyominoWithFrame));
        CHECK_ALLOC(new_data, "failed to reallocate PolyominoWithFrameArray data");
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size] = item;
    arr->size += 1;
    return 0;
}

// Cleanup PolyominoWithFrameArray
void PolyominoWithFrameArray_cleanup(PolyominoWithFrameArray *arr) {
    if (arr && arr->data) {
        for (int i = 0; i < arr->size; i++) {
            CoordinateArray_cleanup(&arr->data[i].shape);
        }
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}