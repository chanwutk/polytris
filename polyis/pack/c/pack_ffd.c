#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#include "utilities.h"
#include "errors.h"


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
int copy_coordinate_array(CoordinateArray *src, CoordinateArray *dest) {
    // Initialize destination array with same capacity as source
    CoordinateArray_init(dest, src->size);

    // Copy all coordinates from source to destination
    for (int i = 0; i < src->size; i++) {
        CoordinateArray_push(dest, src->data[i]);
    }

    return 0;
}

// Comparison function for sorting collage candidates by empty space (descending)
int compare_collage_candidates(const void *a, const void *b) {
    const CollageCandidate *ca = (const CollageCandidate*)a;
    const CollageCandidate *cb = (const CollageCandidate*)b;
    // Sort descending (most empty space first)
    return cb->empty_space - ca->empty_space;
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

                // Assert bounds (should never be out of bounds due to loop constraints)
                ASSERT(IN_BOUNDS(py, px, h, w), "polyomino coordinate out of bounds during placement check");

                // Check collision
                if (GET_TILE(occupied_tiles, h, w, py, px) != 0) {
                    fits = false;
                    break;
                }
            }

            if (fits) {
                // Place the polyomino
                for (int i = 0; i < polyomino_coords->size; i++) {
                    int py = y + polyomino_coords->data[i].y - min_y;
                    int px = x + polyomino_coords->data[i].x - min_x;
                    SET_TILE(occupied_tiles, h, w, py, px, 1);
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
CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w) {
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

            // Copy mask coordinate array
            copy_coordinate_array(&polyomino->mask, &pwf.shape);

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
                qsort(candidates, (size_t)num_candidates, sizeof(CollageCandidate),
                      compare_collage_candidates);
            }

            // Try to place in existing collages
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
                    PolyominoPosition pos = {
                        .oy = oy,
                        .ox = ox,
                        .py = py,
                        .px = px,
                        .rotation = rotation,
                        .frame = frame
                    };

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
            free(candidates);
        }

        if (!placed) {
            // No existing collage could fit this polyomino - create a new collage
            // Create a new empty collage with specified dimensions
            unsigned char *collage = (unsigned char*)calloc((size_t)(h * w), sizeof(unsigned char));
            CHECK_ALLOC(collage, "failed to allocate new collage for packing");

            // Attempt to place the polyomino in the new collage
            Placement placement;
            bool pack_success = try_pack(shape, collage, h, w, &placement);
            ASSERT(pack_success, "failed to pack polyomino in empty collage - this should never happen");

            // Extract position and rotation from successful placement
            int py = placement.y;
            int px = placement.x;
            int rotation = placement.rotation;

            // Create PolyominoPosition
            PolyominoPosition pos = {
                .oy = oy,
                .ox = ox,
                .py = py,
                .px = px,
                .rotation = rotation,
                .frame = frame
            };

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
            UCharPArray_push(&collages_pool, collage);
            CollageArray_push(result, new_collage_positions);

            // Initialize empty space for this new collage
            // Total space minus the polyomino just placed
            int initial_empty_space = (h * w) - polyomino_size;
            IntArray_push(&empty_spaces, initial_empty_space);
        }
    }

    // Cleanup
    UCharPArray_cleanup(&collages_pool);
    IntArray_cleanup(&empty_spaces);
    PolyominoWithFrameArray_cleanup(&all_polyominoes);

    return result;
}
