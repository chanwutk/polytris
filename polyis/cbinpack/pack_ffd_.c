#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#include "utilities_.h"

// ============================================================================
// Macros for Dynamic Array Implementation
// ============================================================================

// Macro to define _init function for simple arrays
#define DEFINE_ARRAY_INIT(TypeName, ElementType) \
int TypeName##_init(TypeName *arr, int initial_capacity) { \
    arr->data = (ElementType*)malloc((size_t)initial_capacity * sizeof(ElementType)); \
    if (!arr->data) return -1; \
    arr->size = 0; \
    arr->capacity = initial_capacity; \
    return 0; \
}

// Macro to define _push function for simple arrays
#define DEFINE_ARRAY_PUSH(TypeName, ElementType) \
int TypeName##_push(TypeName *arr, ElementType value) { \
    if (arr->size >= arr->capacity) { \
        int new_capacity = arr->capacity * 2; \
        ElementType *new_data = (ElementType*)realloc(arr->data, \
                                (size_t)new_capacity * sizeof(ElementType)); \
        if (!new_data) return -1; \
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

// Initialize a UCharPArray
int UCharPArray_init(UCharPArray *arr, int initial_capacity) {
    // Allocate array of pointers
    arr->data = (unsigned char**)malloc((size_t)initial_capacity * sizeof(unsigned char*));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an unsigned char pointer to the array
int UCharPArray_push(UCharPArray *arr, unsigned char *value) {
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
    arr->data = (int*)malloc((size_t)initial_capacity * sizeof(int));
    if (!arr->data) return -1;
    arr->size = 0;
    arr->capacity = initial_capacity;
    return 0;
}

// Push an integer to the array
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

// Cleanup IntArray
void IntArray_cleanup(IntArray *arr) {
    if (arr && arr->data) {
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

// Structure to hold collage candidates with their empty space
typedef struct CollageCandidate {
    int index;        // Index in collages_pool
    int empty_space;  // Amount of empty space in this collage
} CollageCandidate;

// Comparison function for sorting collage candidates by empty space (descending)
int compare_collage_candidates(const void *a, const void *b) {
    const CollageCandidate *ca = (const CollageCandidate*)a;
    const CollageCandidate *cb = (const CollageCandidate*)b;
    // Sort descending (most empty space first)
    return cb->empty_space - ca->empty_space;
}

// ============================================================================
// Polyomino Conversion Helpers
// ============================================================================

// Convert UShortArray mask to CoordinateArray
int convert_mask_to_coordinates(UShortArray *mask, CoordinateArray *coords) {
    // Initialize coordinate array
    int num_pairs = mask->size / 2;
    CoordinateArray_init(coords, num_pairs);

    // Convert coordinate pairs from UShortArray to CoordinateArray
    for (int i = 0; i < num_pairs; i++) {
        Coordinate coord;
        coord.y = (int)mask->data[i * 2];
        coord.x = (int)mask->data[i * 2 + 1];
        CoordinateArray_push(coords, coord);
    }

    return 0;
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

// Try to pack a polyomino (as coordinate array) into the collage
bool try_pack(CoordinateArray *polyomino_coords, unsigned char *occupied_tiles,
              int h, int w, Placement *placement_out) {
    // No rotation in this implementation (rotation = 0)
    int rotation = 0;

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
                placement_out->rotation = rotation;
                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// Main Packing Algorithm
// ============================================================================

// Pack all polyominoes into collages
// Args:
//   polyominoes_arrays: Array of pointers to PolyominoArray
//   num_arrays: Number of arrays in the array
//   h: Height of each collage
//   w: Width of each collage
// Returns:
//   CollageArray containing all packed collages with polyomino positions
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
            convert_mask_to_coordinates(&polyomino->mask, &pwf.shape);

            // Store offset and frame information
            pwf.oy = polyomino->offset_i;
            pwf.ox = polyomino->offset_j;
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
    UCharPArray collages_pool;
    UCharPArray_init(&collages_pool, 16);

    // Storage for empty space tracking (parallel to collages_pool)
    IntArray empty_spaces;
    IntArray_init(&empty_spaces, 16);

    // Process each polyomino in size order (largest first)
    for (int i = 0; i < all_polyominoes.size; i++) {
        PolyominoWithFrame *pwf = &all_polyominoes.data[i];

        // Extract shape, offsets, and frame
        CoordinateArray *shape = &pwf->shape;
        int oy = pwf->oy;
        int ox = pwf->ox;
        int frame = pwf->frame;
        int polyomino_size = shape->size;

        // Build list of collage candidates sorted by empty space (most empty first)
        // Allocate candidates array (max size = number of collages)
        CollageCandidate *candidates = NULL;
        int num_candidates = 0;

        if (collages_pool.size > 0) {
            candidates = (CollageCandidate*)malloc((size_t)collages_pool.size * sizeof(CollageCandidate));
            if (candidates) {
                // Use cached empty space values and filter by polyomino size
                for (int collage_idx = 0; collage_idx < collages_pool.size; collage_idx++) {
                    int empty_space = empty_spaces.data[collage_idx];

                    // Only consider collages with enough empty space
                    if (empty_space >= polyomino_size) {
                        candidates[num_candidates].index = collage_idx;
                        candidates[num_candidates].empty_space = empty_space;
                        num_candidates++;
                    }
                }

                // Sort candidates by empty space (descending order)
                if (num_candidates > 0) {
                    qsort(candidates, (size_t)num_candidates, sizeof(CollageCandidate),
                          compare_collage_candidates);
                }
            }
        }

        // Try to place the polyomino in existing collages (ordered by most empty space first)
        bool placed = false;
        for (int cand_idx = 0; cand_idx < num_candidates; cand_idx++) {
            int collage_idx = candidates[cand_idx].index;
            unsigned char *collage = collages_pool.data[collage_idx];

            // Attempt to pack the polyomino in this collage
            Placement placement;
            if (try_pack(shape, collage, h, w, &placement)) {
                // Successfully placed - extract position and rotation
                int py = placement.y;
                int px = placement.x;
                int rotation = placement.rotation;

                // Create PolyominoPosition
                PolyominoPosition pos;
                pos.oy = oy;
                pos.ox = ox;
                pos.py = py;
                pos.px = px;
                pos.rotation = rotation;
                pos.frame = frame;

                // Copy shape coordinates
                // TODO: Do not copy
                CoordinateArray_init(&pos.shape, shape->size);
                for (int k = 0; k < shape->size; k++) {
                    CoordinateArray_push(&pos.shape, shape->data[k]);
                }

                // Record the polyomino position in this collage
                PolyominoPositionArray_push(&result->data[collage_idx], pos);

                // Update the empty space counter for this collage
                empty_spaces.data[collage_idx] -= polyomino_size;

                placed = true;
                break;
            }
        }

        // Free candidates array
        if (candidates) {
            free(candidates);
            candidates = NULL;
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
                // Extract position and rotation from successful placement
                int py = placement.y;
                int px = placement.x;
                int rotation = placement.rotation;

                // Create PolyominoPosition
                PolyominoPosition pos;
                pos.oy = oy;
                pos.ox = ox;
                pos.py = py;
                pos.px = px;
                pos.rotation = rotation;
                pos.frame = frame;

                // Copy shape coordinates
                // TODO: Do not copy
                CoordinateArray_init(&pos.shape, shape->size);
                for (int k = 0; k < shape->size; k++) {
                    CoordinateArray_push(&pos.shape, shape->data[k]);
                }

                // Create a new positions array for this collage
                PolyominoPositionArray new_collage_positions;
                PolyominoPositionArray_init(&new_collage_positions, 64);
                PolyominoPositionArray_push(&new_collage_positions, pos);

                // Add to collages pool and result
                if (UCharPArray_push(&collages_pool, collage) != 0) {
                    // Push failed - cleanup and return partial result
                    free(collage);
                    CoordinateArray_cleanup(&pos.shape);
                    PolyominoPositionArray_cleanup(&new_collage_positions);
                    break;
                }
                CollageArray_push(result, new_collage_positions);

                // Initialize empty space for this new collage
                // Total space minus the polyomino just placed
                int initial_empty_space = (h * w) - polyomino_size;
                IntArray_push(&empty_spaces, initial_empty_space);
            } else {
                // Should not happen for empty collage, but cleanup if it does
                free(collage);
            }
        }
    }

    // Cleanup
    UCharPArray_cleanup(&collages_pool);
    IntArray_cleanup(&empty_spaces);
    PolyominoWithFrameArray_cleanup(&all_polyominoes);

    return result;
}
