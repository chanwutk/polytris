#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#include "utilities.h"
#include "errors.h"


// ============================================================================
// Enumerations
// ============================================================================

// Packing mode options for bin packing algorithms
typedef enum PackMode {
    Easiest_Fit = 0,  // Pack into collage with most empty space
    First_Fit = 1,    // Pack into first collage that fits
    Best_Fit = 2      // Pack into collage with least empty space that fits
} PackMode;


// ============================================================================
// Helper Macros
// ============================================================================

// Get value from 2D array stored as 1D
#define GET_TILE(tiles, h, w, y, x) ((tiles)[(y) * (w) + (x)])

// Set value in 2D array stored as 1D
#define SET_TILE(tiles, h, w, y, x, value) ((tiles)[(y) * (w) + (x)] = (value))

// Check if coordinate is within bounds
#define IN_BOUNDS(y, x, h, w) ((y) >= 0 && (y) < (h) && (x) >= 0 && (x) < (w))

// ============================================================================
// Polyomino Conversion Helpers
// ============================================================================

// Structure to hold collage candidates with their empty space
typedef struct CollageCandidate {
    int index;        // Index in collages_pool
    int empty_space;  // Amount of empty space in this collage
} CollageCandidate;

// Copy CoordinateArray mask to another CoordinateArray
static inline void copy_coordinate_array(CoordinateArray *src, CoordinateArray *dest) {
    // Initialize destination array with same capacity as source
    CoordinateArray_init(dest, src->size);

    // Copy all coordinates from source to destination
    for (int16_t i = 0; i < src->size; i++) {
        CoordinateArray_push(dest, src->data[i]);
    }
}

// Comparison function for sorting collage candidates by empty space (descending)
static int compare_collage_candidates_descend(const void *a, const void *b) {
    const CollageCandidate *ca = (const CollageCandidate*)a;
    const CollageCandidate *cb = (const CollageCandidate*)b;
    // Sort descending (most empty space first)
    return cb->empty_space - ca->empty_space;
}
// Comparison function for sorting collage candidates by empty space (ascending)
static int compare_collage_candidates_ascend(const void *a, const void *b) {
    const CollageCandidate *ca = (const CollageCandidate*)a;
    const CollageCandidate *cb = (const CollageCandidate*)b;
    // Sort ascending (least empty space first)
    return ca->empty_space - cb->empty_space;
}

// Comparison function for qsort (sort by size descending)
static int compare_polyominoes_by_size(const void *a, const void *b) {
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
    for (int16_t i = 0; i < shape->size; i++) {
        CoordinateArray_push(&pos->shape, shape->data[i]);
    }
}

// Place polyomino at specified position in the occupied tiles grid
// Updates the occupied_tiles array by marking all polyomino coordinates as occupied
static inline void place(CoordinateArray *coords, uint8_t *occupied_tiles,
                         int h, int w, int16_t y, int16_t x) {
    // Mark all polyomino tiles as occupied in the grid
    for (int16_t i = 0; i < coords->size; i++) {
        int16_t py = y + coords->data[i].y;
        int16_t px = x + coords->data[i].x;
        SET_TILE(occupied_tiles, h, w, py, px, 1);
    }
}

// Try to pack a polyomino (as coordinate array) into the collage
// ph and pw are the height and width of the polyomino bounding box
static inline bool try_place(CoordinateArray *coords, uint8_t *occupied_tiles, int h,
                            int w, int16_t ph, int16_t pw, Placement *placement_out) {
    if (coords->size == 0) return false;

    // Try all possible positions where the polyomino would fit
    for (int16_t y = 0; y <= h - ph; y++) {
        for (int16_t x = 0; x <= w - pw; x++) {
            // Check if polyomino fits at this position
            bool fits = true;
            for (int16_t i = 0; i < coords->size; i++) {
                int16_t py = y + coords->data[i].y;
                int16_t px = x + coords->data[i].x;

                // Assert bounds (should never be out of bounds due to loop constraints)
                ASSERT(IN_BOUNDS(py, px, h, w), "polyomino coordinate out of bounds during placement check");

                // Check collision
                if (GET_TILE(occupied_tiles, h, w, py, px) != 0) {
                    fits = false;
                    break;
                }
            }

            if (fits) {
                // Place the polyomino at this position
                place(coords, occupied_tiles, h, w, y, x);

                // Return placement (no adjustment needed since min offsets are 0)
                placement_out->y = y;
                placement_out->x = x;
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
//   mode: Packing mode to use
// Returns:
//   CollageArray containing all packed collages with polyomino positions
CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w, PackMode mode) {
    // Initialize storage for all polyominoes with their frame indices
    PolyominoWithFrameArray all_polyominoes;
    PolyominoWithFrameArray_init(&all_polyominoes, 128);

    // Combine arrays of polyominoes into a single array with frame indices
    for (int array_idx = 0; array_idx < num_arrays; array_idx++) {
        PolyominoArray *array = polyominoes_arrays[array_idx];
        int num_polyominoes = array->size;

        // Convert each polyomino in this array
        for (int poly_idx = 0; poly_idx < num_polyominoes; poly_idx++) {
            Polyomino *polyomino = &array->data[poly_idx];

            // Create PolyominoWithFrame entry
            PolyominoWithFrame pwf;

            // Copy mask coordinate array
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

    // Storage for empty space tracking (parallel to collages_pool)
    IntArray empty_spaces;
    IntArray_init(&empty_spaces, 16);

    // Process each polyomino in size order (largest first)
    for (int i = 0; i < all_polyominoes.size; i++) {
        PolyominoWithFrame *pwf = &all_polyominoes.data[i];

        // Extract shape, offsets, and frame
        CoordinateArray *shape = &pwf->shape;
        int16_t oy = pwf->oy;
        int16_t ox = pwf->ox;
        int32_t frame = pwf->frame;
        int polyomino_size = shape->size;

        // Calculate bounding box dimensions (coordinates are normalized: min=0)
        // Find max_y and max_x to determine height and width
        int16_t max_y = 0;
        int16_t max_x = 0;
        for (int16_t j = 0; j < shape->size; j++) {
            if (shape->data[j].y > max_y) max_y = shape->data[j].y;
            if (shape->data[j].x > max_x) max_x = shape->data[j].x;
        }
        int16_t ph = max_y + 1;
        int16_t pw = max_x + 1;

        // Try to place the polyomino in existing collages (ordered by most empty space first)
        bool placed = false;

        if (collages_pool.size > 0) {
            // Build list of collage candidates sorted by empty space (most empty first)
            CollageCandidate *candidates = (CollageCandidate*)malloc((size_t)collages_pool.size * sizeof(CollageCandidate));
            CHECK_ALLOC(candidates, "failed to allocate candidates array for collage selection");

            int num_candidates = 0;

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
                switch (mode) {
                    case Easiest_Fit:
                        // Sort descending (most empty space first)
                        qsort(candidates, (size_t)num_candidates, sizeof(CollageCandidate),
                              compare_collage_candidates_descend);
                        break;
                    case Best_Fit:
                        // Sort ascending (least empty space first)
                        qsort(candidates, (size_t)num_candidates, sizeof(CollageCandidate),
                              compare_collage_candidates_ascend);
                        break;
                    case First_Fit:
                        // No sorting needed for First Fit
                        break;
                    default:
                        ASSERT(false, "unknown packing mode");
                        break;
                }
            }

            // Try to place in existing collages
            for (int cand_idx = 0; cand_idx < num_candidates; cand_idx++) {
                int collage_idx = candidates[cand_idx].index;
                uint8_t *collage = collages_pool.data[collage_idx];

                // Attempt to pack the polyomino in this collage
                Placement placement;
                if (try_place(shape, collage, h, w, ph, pw, &placement)) {
                    // Successfully placed - create position structure
                    PolyominoPosition pos;
                    create_polyomino_position(&pos, oy, ox, frame, placement, shape);

                    // Record the polyomino position in this collage
                    PolyominoPositionArray_push(&result->data[collage_idx], pos);

                    // Update the empty space counter for this collage
                    empty_spaces.data[collage_idx] -= polyomino_size;

                    placed = true;
                    break;
                }
            }

            // Free candidates array
            free(candidates);
        }

        if (!placed) {
            // No existing collage could fit this polyomino - create a new collage
            // Create a new empty collage with specified dimensions
            uint8_t *collage = (uint8_t*)calloc((size_t)(h * w), sizeof(uint8_t));
            CHECK_ALLOC(collage, "failed to allocate new collage for packing");

            // Place the polyomino at the top-left corner of the new empty collage
            place(shape, collage, h, w, 0, 0);

            // Create placement structure for position tracking
            Placement placement = {.y = 0, .x = 0};

            // Create position structure from successful placement
            PolyominoPosition pos;
            create_polyomino_position(&pos, oy, ox, frame, placement, shape);

            // Create a new positions array for this collage
            PolyominoPositionArray new_collage_positions;
            PolyominoPositionArray_init(&new_collage_positions, 64);
            PolyominoPositionArray_push(&new_collage_positions, pos);

            // Add to collages pool and result
            U8PArray_push(&collages_pool, collage);
            CollageArray_push(result, new_collage_positions);

            // Initialize empty space for this new collage
            // Total space minus the polyomino just placed
            int initial_empty_space = (h * w) - polyomino_size;
            IntArray_push(&empty_spaces, initial_empty_space);
        }
    }

    // Cleanup
    U8PArray_cleanup(&collages_pool);
    IntArray_cleanup(&empty_spaces);
    PolyominoWithFrameArray_cleanup(&all_polyominoes);

    // TODO: Each list of PolyominoPosition should be sorted by frame index for easier processing later
    return result;
}
