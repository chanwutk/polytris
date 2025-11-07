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
#define DEFINE_ARRAY_CLEANUP(TypeName) \
void TypeName##_cleanup(TypeName *arr) { \
    if (arr && arr->data) { \
        free(arr->data); \
        arr->data = NULL; \
    } \
    arr->size = 0; \
    arr->capacity = 0; \
}

// Macro to define complete simple array (init + push + cleanup)
#define DEFINE_ARRAY(TypeName, ElementType) \
DEFINE_ARRAY_INIT(TypeName, ElementType) \
DEFINE_ARRAY_PUSH(TypeName, ElementType) \
DEFINE_ARRAY_CLEANUP(TypeName)

// Macro to define _cleanup function for nested arrays (requires element cleanup)
#define DEFINE_ARRAY_CLEANUP_NESTED(TypeName, ElementCleanupFunc) \
void TypeName##_cleanup(TypeName *arr) { \
    if (arr && arr->data) { \
        for (int i = 0; i < arr->size; i++) { \
            ElementCleanupFunc(&arr->data[i]); \
        } \
        free(arr->data); \
        arr->data = NULL; \
    } \
    arr->size = 0; \
    arr->capacity = 0; \
}

// Macro to define _cleanup function for nested arrays with field-based cleanup
#define DEFINE_ARRAY_CLEANUP_NESTED_WITH_FIELD(TypeName, FieldType, field_name) \
void TypeName##_cleanup(TypeName *arr) { \
    if (arr && arr->data) { \
        for (int i = 0; i < arr->size; i++) { \
            FieldType##_cleanup(&arr->data[i].field_name); \
        } \
        free(arr->data); \
        arr->data = NULL; \
    } \
    arr->size = 0; \
    arr->capacity = 0; \
}

// Macro to define complete nested array (init + push + cleanup with nested cleanup)
#define DEFINE_NESTED_ARRAY(TypeName, ElementType, ElementCleanupFunc) \
DEFINE_ARRAY_INIT(TypeName, ElementType) \
DEFINE_ARRAY_PUSH(TypeName, ElementType) \
DEFINE_ARRAY_CLEANUP_NESTED(TypeName, ElementCleanupFunc)

// Macro to define complete nested array with field-based cleanup
#define DEFINE_NESTED_ARRAY_WITH_FIELD(TypeName, ElementType, FieldType, field_name) \
DEFINE_ARRAY_INIT(TypeName, ElementType) \
DEFINE_ARRAY_PUSH(TypeName, ElementType) \
DEFINE_ARRAY_CLEANUP_NESTED_WITH_FIELD(TypeName, FieldType, field_name)

// Macro to define _cleanup function for pointer arrays (frees stored pointers)
#define DEFINE_ARRAY_CLEANUP_POINTERS(TypeName) \
void TypeName##_cleanup(TypeName *arr) { \
    if (arr && arr->data) { \
        for (int i = 0; i < arr->size; i++) { \
            if (arr->data[i]) { \
                free(arr->data[i]); \
            } \
        } \
        free(arr->data); \
        arr->data = NULL; \
    } \
    arr->size = 0; \
    arr->capacity = 0; \
}

// Macro to define complete pointer array (init + push + cleanup with pointer cleanup)
#define DEFINE_POINTER_ARRAY(TypeName, ElementType) \
DEFINE_ARRAY_INIT(TypeName, ElementType) \
DEFINE_ARRAY_PUSH(TypeName, ElementType) \
DEFINE_ARRAY_CLEANUP_POINTERS(TypeName)


// ============================================================================
// Structure Definitions
// ============================================================================

typedef struct ShortArray {
    short *data;
    int size;
    int capacity;
} ShortArray;
// Generate ShortArray functions: init, push, cleanup
DEFINE_ARRAY(ShortArray, short)

// Represents a 2D coordinate/point
typedef struct Coordinate {
    short y;
    short x;
} Coordinate;

// Dynamic array of coordinates
typedef struct CoordinateArray {
    Coordinate *data;
    int size;
    int capacity;
} CoordinateArray;
// Generate CoordinateArray functions: init, push, cleanup
DEFINE_ARRAY(CoordinateArray, Coordinate)

// Polyomino structure with coordinate-based mask
typedef struct Polyomino {
    CoordinateArray mask;
    int offset_i;
    int offset_j;
} Polyomino;

typedef struct PolyominoArray {
    Polyomino *data;
    int size;
    int capacity;
} PolyominoArray;
// Generate PolyominoArray with automatic Polyomino cleanup (cleans mask field)
DEFINE_NESTED_ARRAY_WITH_FIELD(PolyominoArray, Polyomino, CoordinateArray, mask)

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
// Generate PolyominoPositionArray with automatic PolyominoPosition cleanup (cleans shape field)
DEFINE_NESTED_ARRAY_WITH_FIELD(PolyominoPositionArray, PolyominoPosition, CoordinateArray, shape)

// List of collages (each collage contains multiple polyomino positions)
typedef struct CollageArray {
    PolyominoPositionArray *data;
    int size;
    int capacity;
} CollageArray;
// Generate CollageArray functions: init, push, cleanup with nested cleanup
DEFINE_NESTED_ARRAY(CollageArray, PolyominoPositionArray, PolyominoPositionArray_cleanup)

// Dynamic array of unsigned char pointers (for collage occupied tiles pool)
typedef struct UCharPArray {
    unsigned char **data;  // Array of unsigned char pointers
    int size;              // Current number of elements
    int capacity;          // Allocated capacity
} UCharPArray;
// Generate UCharPArray functions: init, push, cleanup with pointer cleanup
DEFINE_POINTER_ARRAY(UCharPArray, unsigned char*)

typedef struct IntArray {
    int *data;      // Array of integers
    int size;       // Current number of elements
    int capacity;   // Allocated capacity
} IntArray;
// Generate IntArray functions: init, push, cleanup
DEFINE_ARRAY(IntArray, int)

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

// Generate PolyominoWithFrameArray with automatic PolyominoWithFrame cleanup (cleans shape field)
DEFINE_NESTED_ARRAY_WITH_FIELD(PolyominoWithFrameArray, PolyominoWithFrame, CoordinateArray, shape)