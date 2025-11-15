/**
 * @file pack_bfd.c
 * @brief Implementation of Best-Fit-Descending (BFD) polyomino packing algorithm
 *
 * This file implements a sophisticated packing algorithm that places polyominoes
 * into fixed-size collages by finding the best-fit position in each collage.
 * The algorithm maintains metadata about unoccupied regions to optimize placement
 * decisions and minimize fragmentation.
 *
 * Key features:
 * - Best-fit placement strategy for tight packing
 * - Region-based empty space tracking
 * - Support for multiple collages when items don't fit
 * - Efficient collision detection using occupied tile bitmaps
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#include "utilities.h"
#include "errors.h"

// ============================================================================
// Macros for Dynamic Array Implementation
// ============================================================================

/**
 * @brief Macro to generate init function for dynamic arrays
 *
 * Generates a TypeName_init function that allocates initial capacity
 * for a dynamic array structure.
 */
#define DEFINE_ARRAY_INIT(TypeName, ElementType) \
int TypeName##_init(TypeName *arr, int initial_capacity) { \
    CHECK_NULL(arr, #TypeName " array pointer is NULL"); \
    ASSERT(initial_capacity > 0, EXIT_ERROR_INVALID_ARGUMENT, \
                    "initial_capacity must be positive"); \
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

// Dynamic array of coordinate arrays (for unoccupied regions)
typedef struct CoordinateArrayList {
    CoordinateArray *data;
    int size;
    int capacity;
} CoordinateArrayList;

// Dynamic array of integers (for region sizes)
typedef struct IntArray {
    int *data;
    int size;
    int capacity;
} IntArray;

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

// Metadata for a collage with cached unoccupied regions
typedef struct CollageMetadata {
    unsigned char *occupied_tiles;  // 2D array stored as 1D (h*w)
    int height;
    int width;
    CoordinateArrayList unoccupied_spaces;  // List of unoccupied regions
    IntArray space_sizes;               // Size of each region
} CollageMetadata;

// Dynamic array of CollageMetadata
typedef struct CollageMetadataArray {
    CollageMetadata *data;
    int size;
    int capacity;
} CollageMetadataArray;

// Dynamic array of unsigned char pointers (for collage occupied tiles pool)
typedef struct U8PArray {
    unsigned char **data;  // Array of unsigned char pointers
    int size;              // Current number of elements
    int capacity;          // Allocated capacity
} U8PArray;

// ============================================================================
// Memory Management Functions for Dynamic Arrays
// ============================================================================

// Initialize a coordinate array
int CoordinateArray_init(CoordinateArray *arr, int initial_capacity) {
    arr->data = (Coordinate*)malloc((size_t)initial_capacity * sizeof(Coordinate));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push a coordinate to the array
int CoordinateArray_push(CoordinateArray *arr, Coordinate coord) {
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        Coordinate *new_data = (Coordinate*)realloc(arr->data,
                                                     (size_t)new_capacity * sizeof(Coordinate));
        if (!new_data) return -1;
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

// Initialize a coordinate array list
int CoordinateArrayList_init(CoordinateArrayList *list, int initial_capacity) {
    list->data = (CoordinateArray*)malloc((size_t)initial_capacity * sizeof(CoordinateArray));
    if (!list->data) return -1;
    list->size = 0;
    list->capacity = initial_capacity;
    return 0;
}

// Push a coordinate array to the list
int CoordinateArrayList_push(CoordinateArrayList *list, CoordinateArray arr) {
    // Expand if necessary
    if (list->size >= list->capacity) {
        int new_capacity = list->capacity * 2;
        CoordinateArray *new_data = (CoordinateArray*)realloc(list->data,
                                                               (size_t)new_capacity * sizeof(CoordinateArray));
        if (!new_data) return -1;
        list->data = new_data;
        list->capacity = new_capacity;
    }
    // Push the coordinate array
    list->data[list->size] = arr;
    list->size += 1;
    return 0;
}

// Cleanup coordinate array list
void CoordinateArrayList_cleanup(CoordinateArrayList *list) {
    if (list && list->data) {
        // Cleanup each coordinate array
        for (int i = 0; i < list->size; i++) {
            CoordinateArray_cleanup(&list->data[i]);
        }
        free(list->data);
        list->data = NULL;
    }
    list->size = 0;
    list->capacity = 0;
}

// Initialize an integer size array
int IntArray_init(IntArray *arr, int initial_capacity) {
    arr->data = (int*)malloc((size_t)initial_capacity * sizeof(int));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an integer to the size array
int IntArray_push(IntArray *arr, int value) {
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        int *new_data = (int*)realloc(arr->data, (size_t)new_capacity * sizeof(int));
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the value
    arr->data[arr->size] = value;
    arr->size += 1;
    return 0;
}

// Cleanup integer size array
void IntArray_cleanup(IntArray *arr) {
    if (arr && arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// Initialize a PolyominoPositionArray
int PolyominoPositionArray_init(PolyominoPositionArray *arr, int initial_capacity) {
    arr->data = (PolyominoPosition*)malloc((size_t)initial_capacity * sizeof(PolyominoPosition));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push a PolyominoPosition to the array
int PolyominoPositionArray_push(PolyominoPositionArray *arr, PolyominoPosition pos) {
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        PolyominoPosition *new_data = (PolyominoPosition*)realloc(arr->data,
                                                                   (size_t)new_capacity * sizeof(PolyominoPosition));
        if (!new_data) return -1;
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
    list->data = (PolyominoPositionArray*)malloc((size_t)initial_capacity * sizeof(PolyominoPositionArray));
    if (!list->data) return -1;
    list->size = 0;
    list->capacity = initial_capacity;
    return 0;
}

// Push a PolyominoPositionArray to the list
int CollageArray_push(CollageArray *list, PolyominoPositionArray arr) {
    // Expand if necessary
    if (list->size >= list->capacity) {
        int new_capacity = list->capacity * 2;
        PolyominoPositionArray *new_data = (PolyominoPositionArray*)realloc(list->data,
                                                                             (size_t)new_capacity * sizeof(PolyominoPositionArray));
        if (!new_data) return -1;
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

// Initialize a CollageMetadataArray
int CollageMetadataArray_init(CollageMetadataArray *arr, int initial_capacity) {
    arr->data = (CollageMetadata*)malloc((size_t)initial_capacity * sizeof(CollageMetadata));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push a CollageMetadata to the array
int CollageMetadataArray_push(CollageMetadataArray *arr, CollageMetadata meta) {
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        CollageMetadata *new_data = (CollageMetadata*)realloc(arr->data,
                                                               (size_t)new_capacity * sizeof(CollageMetadata));
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the metadata
    arr->data[arr->size] = meta;
    arr->size += 1;
    return 0;
}

// Cleanup CollageMetadataArray
void CollageMetadataArray_cleanup(CollageMetadataArray *arr) {
    if (arr && arr->data) {
        // Cleanup each collage metadata
        for (int i = 0; i < arr->size; i++) {
            if (arr->data[i].occupied_tiles) {
                free(arr->data[i].occupied_tiles);
            }
            CoordinateArrayList_cleanup(&arr->data[i].unoccupied_spaces);
            IntArray_cleanup(&arr->data[i].space_sizes);
        }
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

// Initialize a U8PArray
int U8PArray_init(U8PArray *arr, int initial_capacity) {
    // Allocate array of pointers
    arr->data = (unsigned char**)malloc((size_t)initial_capacity * sizeof(unsigned char*));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an unsigned char pointer to the array
int U8PArray_push(U8PArray *arr, unsigned char *value) {
    // Expand if necessary
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        unsigned char **new_data = (unsigned char**)realloc(arr->data,
                                                            (size_t)new_capacity * sizeof(unsigned char*));
        if (!new_data) return -1;
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    // Push the pointer
    arr->data[arr->size] = value;
    arr->size += 1;
    return 0;
}

// Cleanup U8PArray (two-level cleanup: frees stored pointers then array)
void U8PArray_cleanup(U8PArray *arr) {
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
// Helper Functions
// ============================================================================

// Get value from 2D array stored as 1D
static inline unsigned char get_tile(unsigned char *tiles, int h, int w, int y, int x) {
    return tiles[y * w + x];
}

// Set value in 2D array stored as 1D
static inline void set_tile(unsigned char *tiles, int h, int w, int y, int x, unsigned char value) {
    tiles[y * w + x] = value;
}

// Check if coordinate is within bounds
static inline bool in_bounds(int y, int x, int h, int w) {
    return y >= 0 && y < h && x >= 0 && x < w;
}

// ============================================================================
// Connected Component Labeling using Flood Fill
// ============================================================================

// Recursive flood fill helper
void flood_fill_recursive(unsigned char *tiles, int *labels, int h, int w,
                         int y, int x, int label, CoordinateArray *region) {
    // Check bounds
    if (!in_bounds(y, x, h, w)) return;

    // Check if already labeled or occupied
    if (labels[y * w + x] != 0) return;
    if (get_tile(tiles, h, w, y, x) != 0) return;

    // Label this cell
    labels[y * w + x] = label;

    // Add to region
    Coordinate coord = {y, x};
    CoordinateArray_push(region, coord);

    // Flood fill neighbors (4-connectivity)
    flood_fill_recursive(tiles, labels, h, w, y - 1, x, label, region);
    flood_fill_recursive(tiles, labels, h, w, y + 1, x, label, region);
    flood_fill_recursive(tiles, labels, h, w, y, x - 1, label, region);
    flood_fill_recursive(tiles, labels, h, w, y, x + 1, label, region);
}

// Extract unoccupied connected regions as coordinate arrays
int extract_unoccupied_spaces(unsigned char *occupied_tiles, int h, int w,
                              CoordinateArrayList *unoccupied_spaces,
                              IntArray *space_sizes) {
    // Allocate label array
    int *labels = (int*)calloc((size_t)(h * w), sizeof(int));
    if (!labels) return -1;

    int current_label = 1;

    // Scan for unoccupied cells and flood fill
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // If cell is empty and not yet labeled
            if (get_tile(occupied_tiles, h, w, y, x) == 0 && labels[y * w + x] == 0) {
                // Create new region
                CoordinateArray region;
                CoordinateArray_init(&region, 64);

                // Flood fill to find all connected cells
                flood_fill_recursive(occupied_tiles, labels, h, w, y, x, current_label, &region);

                // Add region to list
                CoordinateArrayList_push(unoccupied_spaces, region);
                IntArray_push(space_sizes, region.size);

                current_label++;
            }
        }
    }

    // Cleanup
    free(labels);
    return 0;
}

// ============================================================================
// Collage Update After Placement
// ============================================================================

// Update collage metadata after placing a polyomino
int update_collage_after_placement(CollageMetadata *collage_meta,
                                   Placement placement,
                                   CoordinateArray *polyomino_shape) {
    int py = placement.y;
    int px = placement.x;

    // Find which unoccupied space contains the placement
    // Use the first tile of the polyomino shape
    if (polyomino_shape->size == 0) return -1;

    Coordinate first_tile = polyomino_shape->data[0];
    int tile_abs_y = py + first_tile.y;
    int tile_abs_x = px + first_tile.x;

    // Find the affected space
    int affected_space_idx = -1;
    for (int i = 0; i < collage_meta->unoccupied_spaces.size; i++) {
        CoordinateArray *space = &collage_meta->unoccupied_spaces.data[i];
        // Check if any coordinate in this space matches our tile
        for (int j = 0; j < space->size; j++) {
            if (space->data[j].y == tile_abs_y && space->data[j].x == tile_abs_x) {
                affected_space_idx = i;
                break;
            }
        }
        if (affected_space_idx != -1) break;
    }

    if (affected_space_idx == -1) return -1;

    // Mark placed tiles as occupied in occupied_tiles
    for (int i = 0; i < polyomino_shape->size; i++) {
        int y = py + polyomino_shape->data[i].y;
        int x = px + polyomino_shape->data[i].x;
        set_tile(collage_meta->occupied_tiles, collage_meta->height, collage_meta->width, y, x, 1);
    }

    // Get affected space
    CoordinateArray affected_space = collage_meta->unoccupied_spaces.data[affected_space_idx];

    // Create a temporary occupied tiles array for the remaining space
    // We'll mark the placed polyomino tiles to subtract them
    unsigned char *temp_tiles = (unsigned char*)calloc((size_t)(collage_meta->height * collage_meta->width),
                                                        sizeof(unsigned char));
    if (!temp_tiles) return -1;

    // Mark the affected space in temp_tiles
    for (int i = 0; i < affected_space.size; i++) {
        int y = affected_space.data[i].y;
        int x = affected_space.data[i].x;
        set_tile(temp_tiles, collage_meta->height, collage_meta->width, y, x, 0);
    }

    // Mark placed polyomino as occupied (inverted logic for flood fill)
    for (int i = 0; i < polyomino_shape->size; i++) {
        int y = py + polyomino_shape->data[i].y;
        int x = px + polyomino_shape->data[i].x;
        set_tile(temp_tiles, collage_meta->height, collage_meta->width, y, x, 1);
    }

    // Extract new connected components from remaining space
    CoordinateArrayList new_spaces;
    IntArray new_space_sizes;
    CoordinateArrayList_init(&new_spaces, 4);
    IntArray_init(&new_space_sizes, 4);

    extract_unoccupied_spaces(temp_tiles, collage_meta->height, collage_meta->width,
                             &new_spaces, &new_space_sizes);

    // Remove old affected space and insert new spaces
    // First, cleanup the old affected space
    CoordinateArray_cleanup(&collage_meta->unoccupied_spaces.data[affected_space_idx]);

    // Remove from arrays by shifting
    for (int i = affected_space_idx; i < collage_meta->unoccupied_spaces.size - 1; i++) {
        collage_meta->unoccupied_spaces.data[i] = collage_meta->unoccupied_spaces.data[i + 1];
        collage_meta->space_sizes.data[i] = collage_meta->space_sizes.data[i + 1];
    }
    collage_meta->unoccupied_spaces.size--;
    collage_meta->space_sizes.size--;

    // Insert new spaces
    for (int i = 0; i < new_spaces.size; i++) {
        CoordinateArrayList_push(&collage_meta->unoccupied_spaces, new_spaces.data[i]);
        IntArray_push(&collage_meta->space_sizes, new_space_sizes.data[i]);
    }

    // Cleanup (don't cleanup individual arrays as they were moved)
    free(new_spaces.data);
    free(new_space_sizes.data);
    free(temp_tiles);

    return 0;
}

// ============================================================================
// Region Counting
// ============================================================================

// Count regions with size >= min_size
int count_regions_at_least(CollageMetadata *collage_meta, int min_size) {
    int count = 0;
    for (int i = 0; i < collage_meta->space_sizes.size; i++) {
        if (collage_meta->space_sizes.data[i] >= min_size) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// Polyomino Packing
// ============================================================================

// Try to pack a polyomino into the collage
// Returns true if successful, fills placement_out
bool try_pack(CoordinateArray *polyomino_shape, unsigned char *occupied_tiles,
             int h, int w, Placement *placement_out) {
    // No rotation in this implementation
    // Find bounding box of polyomino
    if (polyomino_shape->size == 0) return false;

    int min_y = polyomino_shape->data[0].y;
    int max_y = polyomino_shape->data[0].y;
    int min_x = polyomino_shape->data[0].x;
    int max_x = polyomino_shape->data[0].x;

    for (int i = 1; i < polyomino_shape->size; i++) {
        if (polyomino_shape->data[i].y < min_y) min_y = polyomino_shape->data[i].y;
        if (polyomino_shape->data[i].y > max_y) max_y = polyomino_shape->data[i].y;
        if (polyomino_shape->data[i].x < min_x) min_x = polyomino_shape->data[i].x;
        if (polyomino_shape->data[i].x > max_x) max_x = polyomino_shape->data[i].x;
    }

    int ph = max_y - min_y + 1;
    int pw = max_x - min_x + 1;

    // Try all possible positions
    for (int y = 0; y <= h - ph; y++) {
        for (int x = 0; x <= w - pw; x++) {
            // Check if polyomino fits at this position
            bool fits = true;
            for (int i = 0; i < polyomino_shape->size; i++) {
                int py = y + polyomino_shape->data[i].y;
                int px = x + polyomino_shape->data[i].x;

                // Check bounds
                if (!in_bounds(py, px, h, w)) {
                    fits = false;
                    break;
                }

                // Check collision
                if (get_tile(occupied_tiles, h, w, py, px) != 0) {
                    fits = false;
                    break;
                }
            }

            if (fits) {
                // Place the polyomino
                for (int i = 0; i < polyomino_shape->size; i++) {
                    int py = y + polyomino_shape->data[i].y;
                    int px = x + polyomino_shape->data[i].x;
                    set_tile(occupied_tiles, h, w, py, px, 1);
                }

                // Return placement
                placement_out->y = y;
                placement_out->x = x;
                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// Polyomino Conversion Helpers
// ============================================================================

// Copy CoordinateArray mask to another CoordinateArray
static inline void copy_coordinate_array(CoordinateArray *src, CoordinateArray *dest) {
    // Initialize destination array with same capacity as source
    CoordinateArray_init(dest, src->size);

    // Copy all coordinates from source to destination
    for (int16_t i = 0; i < src->size; i++) {
        CoordinateArray_push(dest, src->data[i]);
    }
}

// Get polyomino size from coordinate array
int get_polyomino_size(CoordinateArray *coords) {
    return coords->size;
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
    arr->data = (PolyominoWithFrame*)malloc((size_t)initial_capacity * sizeof(PolyominoWithFrame));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push to PolyominoWithFrameArray
int PolyominoWithFrameArray_push(PolyominoWithFrameArray *arr, PolyominoWithFrame item) {
    if (arr->size >= arr->capacity) {
        int new_capacity = arr->capacity * 2;
        PolyominoWithFrame *new_data = (PolyominoWithFrame*)realloc(arr->data,
                                                                     (size_t)new_capacity * sizeof(PolyominoWithFrame));
        if (!new_data) return -1;
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

// Comparison function for qsort (sort by size descending)
int compare_polyominoes_by_size(const void *a, const void *b) {
    const PolyominoWithFrame *pa = (const PolyominoWithFrame*)a;
    const PolyominoWithFrame *pb = (const PolyominoWithFrame*)b;
    // Sort descending (larger first)
    return pb->size - pa->size;
}

// ============================================================================
// Try Pack with Coordinate Arrays
// ============================================================================

// Create a PolyominoPosition from placement result
// This helper function encapsulates the common logic of creating a position
// structure from a successful placement attempt
static inline void create_polyomino_position(
    PolyominoPosition *pos,
    int oy, int ox, int frame,
    Placement placement,
    const CoordinateArray *shape
) {
    // Initialize position with placement data
    pos->oy = oy;
    pos->ox = ox;
    pos->py = placement.y;
    pos->px = placement.x;
    pos->frame = frame;

    // Copy shape coordinates
    // TODO: Do not copy
    CoordinateArray_init(&pos->shape, shape->size);
    for (int k = 0; k < shape->size; k++) {
        CoordinateArray_push(&pos->shape, shape->data[k]);
    }
}

// Try to pack a polyomino (as coordinate array) into the collage
bool try_pack(CoordinateArray *polyomino_coords, unsigned char *occupied_tiles,
                     int h, int w, Placement *placement_out) {
    // No rotation in this implementation
    // Find bounding box of polyomino
    if (polyomino_coords->size == 0) return false;

    int min_y = polyomino_coords->data[0].y;
    int max_y = polyomino_coords->data[0].y;
    int min_x = polyomino_coords->data[0].x;
    int max_x = polyomino_coords->data[0].x;

    for (int i = 1; i < polyomino_coords->size; i++) {
        if (polyomino_coords->data[i].y < min_y) min_y = polyomino_coords->data[i].y;
        if (polyomino_coords->data[i].y > max_y) max_y = polyomino_coords->data[i].y;
        if (polyomino_coords->data[i].x < min_x) min_x = polyomino_coords->data[i].x;
        if (polyomino_coords->data[i].x > max_x) max_x = polyomino_coords->data[i].x;
    }

    int ph = max_y - min_y + 1;
    int pw = max_x - min_x + 1;

    // Try all possible positions where the polyomino would fit
    for (int y = 0; y <= h - ph; y++) {
        for (int x = 0; x <= w - pw; x++) {
            // Check if polyomino fits at this position
            bool fits = true;
            for (int i = 0; i < polyomino_coords->size; i++) {
                int py = y + polyomino_coords->data[i].y - min_y;
                int px = x + polyomino_coords->data[i].x - min_x;

                // // Check bounds
                // if (!in_bounds(py, px, h, w)) {
                //     fits = false;
                //     break;
                // }

                // Check collision
                if (get_tile(occupied_tiles, h, w, py, px) != 0) {
                    fits = false;
                    break;
                }
            }

            if (fits) {
                // Place the polyomino
                for (int i = 0; i < polyomino_coords->size; i++) {
                    int py = y + polyomino_coords->data[i].y - min_y;
                    int px = x + polyomino_coords->data[i].x - min_x;
                    set_tile(occupied_tiles, h, w, py, px, 1);
                }

                // Return placement (adjust for min offset)
                placement_out->y = y - min_y;
                placement_out->x = x - min_x;
                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// Main Packing Algorithm
// ============================================================================

/**
 * @brief Pack all polyominoes into fixed-size collages using Best-Fit-Descending
 *
 * This is the main entry point for the BFD packing algorithm. It collects all
 * polyominoes from multiple arrays, sorts them by size (largest first), and
 * places them into collages using a best-fit strategy that minimizes fragmentation.
 *
 * Algorithm overview:
 * 1. Collect all polyominoes from input arrays and tag with frame indices
 * 2. Sort polyominoes by size (descending) for optimal packing
 * 3. For each polyomino, try to place it in existing collages using best-fit
 * 4. Create new collages as needed when no existing collage has space
 * 5. Update occupied tile bitmaps and unoccupied region metadata after each placement
 *
 * @param polyominoes_arrays Array of pointers to PolyominoArray structures
 * @param num_arrays Number of PolyominoArray pointers in the array
 * @param h Height of each collage in tiles
 * @param w Width of each collage in tiles
 *
 * @return Pointer to a newly allocated CollageArray containing all packed collages
 *
 * @note Polyominoes are copied into the result, originals are not modified
 * @note Caller must free the result using CollageArray_cleanup() and free()
 * @note The algorithm may create multiple collages if needed
 */
CollageArray* pack_all(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w) {
    // Initialize storage for all polyominoes with their frame indices
    PolyominoWithFrameArray all_polyominoes;
    PolyominoWithFrameArray_init(&all_polyominoes, 256);

    // Combine arrays of polyominoes into a single array with frame indices
    for (int array_idx = 0; array_idx < num_arrays; array_idx++) {
        PolyominoArray *array = polyominoes_arrays[array_idx];
        int num_polyominoes = array->size;

        // Convert each polyomino in this array
        for (int poly_idx = 0; poly_idx < num_polyominoes; poly_idx++) {
            Polyomino *polyomino = &array->data[poly_idx];

            // Create PolyominoWithFrame entry
            PolyominoWithFrame pwf;

            // Convert mask to coordinate array
            copy_coordinate_array(&polyomino->mask, &pwf.shape);

            // Store offset and frame information
            pwf.oy = polyomino->offset_y;
            pwf.ox = polyomino->offset_x;
            pwf.frame = array_idx;
            pwf.size = pwf.shape.size;

            // Add to array
            PolyominoWithFrameArray_push(&all_polyominoes, pwf);
        }
    }

    // If no polyominoes, return empty result
    if (all_polyominoes.size == 0) {
        PolyominoWithFrameArray_cleanup(&all_polyominoes);
        CollageArray *result = (CollageArray*)malloc(sizeof(CollageArray));
        CollageArray_init(result, 1);
        return result;
    }

    // Sort polyominoes by size (largest first) for better packing efficiency
    qsort(all_polyominoes.data, (size_t)all_polyominoes.size,
          sizeof(PolyominoWithFrame), compare_polyominoes_by_size);

    // Initialize storage for collages and their corresponding polyomino positions
    CollageArray *result = (CollageArray*)malloc(sizeof(CollageArray));
    CollageArray_init(result, 16);

    // Storage for collage occupied tiles arrays
    U8PArray collages_pool;
    U8PArray_init(&collages_pool, 16);

    // Process each polyomino in size order (largest first)
    for (int i = 0; i < all_polyominoes.size; i++) {
        PolyominoWithFrame *pwf = &all_polyominoes.data[i];

        // Extract shape, offsets, and frame
        CoordinateArray *shape = &pwf->shape;
        int oy = pwf->oy;
        int ox = pwf->ox;
        int frame = pwf->frame;

        // Try to place the polyomino in an existing collage
        bool placed = false;
        for (int collage_idx = 0; collage_idx < collages_pool.size; collage_idx++) {
            unsigned char *collage = collages_pool.data[collage_idx];

            // Attempt to pack the polyomino in this collage
            Placement placement;
            if (try_pack(shape, collage, h, w, &placement)) {
                // Successfully placed - create position structure
                PolyominoPosition pos;
                create_polyomino_position(&pos, oy, ox, frame, placement, shape);

                // Record the polyomino position in this collage
                PolyominoPositionArray_push(&result->data[collage_idx], pos);

                placed = true;
                break;
            }
        }

        if (!placed) {
            // No existing collage could fit this polyomino - create a new collage
            // Create a new empty collage with specified dimensions
            unsigned char *collage = (unsigned char*)calloc((size_t)(h * w), sizeof(unsigned char));
            if (!collage) {
                // Cleanup and return partial result
                break;
            }

            // Attempt to place the polyomino in the new collage
            Placement placement;
            if (try_pack(shape, collage, h, w, &placement)) {
                // Create position structure from successful placement
                PolyominoPosition pos;
                create_polyomino_position(&pos, oy, ox, frame, placement, shape);

                // Create a new positions array for this collage
                PolyominoPositionArray new_collage_positions;
                PolyominoPositionArray_init(&new_collage_positions, 64);
                PolyominoPositionArray_push(&new_collage_positions, pos);

                // Add to collages pool and result
                if (U8PArray_push(&collages_pool, collage) != 0) {
                    // Push failed - cleanup and return partial result
                    free(collage);
                    CoordinateArray_cleanup(&pos.shape);
                    PolyominoPositionArray_cleanup(&new_collage_positions);
                    break;
                }
                CollageArray_push(result, new_collage_positions);
            } else {
                // Should not happen for empty collage, but cleanup if it does
                free(collage);
            }
        }
    }

    // Cleanup
    U8PArray_cleanup(&collages_pool);
    PolyominoWithFrameArray_cleanup(&all_polyominoes);

    return result;
}

// ============================================================================
// Cleanup
// ============================================================================

void CollageMetadata_cleanup(CollageMetadata *meta) {
    if (meta) {
        if (meta->occupied_tiles) {
            free(meta->occupied_tiles);
            meta->occupied_tiles = NULL;
        }
        CoordinateArrayList_cleanup(&meta->unoccupied_spaces);
        IntArray_cleanup(&meta->space_sizes);
    }
}
