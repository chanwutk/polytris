/**
 * @file group_tiles.c
 * @brief Implementation of tile grouping and polyomino extraction
 *
 * This file implements connected component analysis using flood-fill to extract
 * polyominoes from a binary bitmap. The algorithm uses 4-connectivity (orthogonal
 * neighbors only) and supports optional padding modes.
 */

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "utilities.h"
#include "errors.h"

/**
 * @brief Direction arrays for 4-connectivity movement
 *
 * These arrays define the four orthogonal directions: up, left, down, right.
 * Used for flood-fill traversal and padding operations.
 */
static const int16_t DIRECTIONS_Y[4] = {-1, 0, 1, 0};  /**< Y-offsets for 4 directions */
static const int16_t DIRECTIONS_X[4] = {0, -1, 0, 1};  /**< X-offsets for 4 directions */

/**
 * @brief Comparison function for qsort to sort polyominoes by size (descending)
 *
 * This comparator sorts polyominoes in descending order by the number of tiles
 * they contain (mask size). Used to optimize packing by processing larger
 * polyominoes first.
 *
 * @param a Pointer to first Polyomino (as void* for qsort compatibility)
 * @param b Pointer to second Polyomino (as void* for qsort compatibility)
 * @return Negative if a > b (a comes first), positive if b > a, 0 if equal
 */
static int compare_polyomino_by_mask_length(const void *a, const void *b) {
    const Polyomino *poly_a = (const Polyomino *)a;
    const Polyomino *poly_b = (const Polyomino *)b;
    // Compare by mask length (size field of CoordinateArray) in descending order
    // Larger masks first (negative return means a comes before b)
    return poly_b->mask.size - poly_a->mask.size;
}

/**
 * @brief Find all tiles connected to a starting position using flood-fill
 *
 * This function performs an iterative flood-fill starting from the given position
 * to identify all connected tiles in the bitmap. It uses a stack-based approach
 * (not recursion) to avoid stack overflow on large regions. The bitmap array is
 * modified in-place to mark visited tiles.
 *
 * @param bitmap Working array of group IDs (modified in-place to mark visited tiles)
 * @param bitmap_input Original binary bitmap indicating occupied tiles
 * @param h Height of the bitmap
 * @param w Width of the bitmap
 * @param start_y Starting Y-coordinate for flood-fill
 * @param start_x Starting X-coordinate for flood-fill
 * @param mode Padding mode that affects connectivity rules (see group_tiles for details)
 *
 * @return CoordinateArray containing all coordinates in the connected component
 *
 * @note The bitmap parameter is modified to prevent revisiting tiles
 * @note Uses 4-connectivity (orthogonal neighbors only)
 * @note Caller must free the returned CoordinateArray
 */
static CoordinateArray find_connected_tiles(
    int16_t *bitmap,
    uint8_t *bitmap_input,
    int16_t h,
    int16_t w,
    int16_t start_y,
    int16_t start_x,
    int8_t mode
) {
    int16_t value = bitmap[start_y * w + start_x];

    // Initialize arrays
    CoordinateArray filled, stack;
    CoordinateArray_init(&filled, 16);
    CoordinateArray_init(&stack, 16);

    // Push initial coordinates
    CoordinateArray_push(&stack, (Coordinate){.y = start_y, .x = start_x});

    // Flood fill algorithm
    while (stack.size > 0) {
        // Pop coordinates from stack
        stack.size--;
        int16_t x = stack.data[stack.size].x;
        int16_t y = stack.data[stack.size].y;

        if (bitmap[y * w + x] == value && (x != start_x || y != start_y)) {
            continue;  // Already visited
        }

        // Mark current position as visited and add to result
        bitmap[y * w + x] = value;
        CoordinateArray_push(&filled, (Coordinate){.y = y, .x = x});

        uint8_t curr_occupancy = bitmap_input[y * w + x];

        // Check all 4 directions for unvisited connected tiles
        for (int16_t i = 0; i < 4; i++) {
            int16_t yy = y + DIRECTIONS_Y[i];
            int16_t xx = x + DIRECTIONS_X[i];

            // Check bounds
            if (0 <= yy && yy < h && 0 <= xx && xx < w) {
                int16_t next_group = bitmap[yy * w + xx];

                // Add neighbors that are non-zero and different from current value
                // (meaning they haven't been visited yet)
                if (next_group != 0 && next_group != value) {
                    // Check padding mode conditions
                    if (mode != 1 || curr_occupancy == 1 || bitmap_input[yy * w + xx] == 1) {
                        CoordinateArray_push(&stack, (Coordinate){.y = yy, .x = xx});
                    }
                }
            }
        }
    }

    // Free the array's data before returning
    CoordinateArray_cleanup(&stack);
    return filled;
}

/**
 * @brief Add padding tiles around occupied positions in the bitmap
 *
 * This function adds padding by marking empty neighbors of occupied tiles
 * with the value 2. Padding helps connect nearby polyominoes or ensures
 * isolated tiles have a minimum bounding box size.
 *
 * @param bitmap Binary bitmap to modify (0=empty, 1=occupied, 2=padding after execution)
 * @param h Height of the bitmap
 * @param w Width of the bitmap
 *
 * @note The bitmap is modified in-place
 * @note Only orthogonal neighbors (4-connectivity) receive padding
 * @note Padding tiles are marked with value 2 to distinguish from original tiles
 */
static inline void add_padding(uint8_t *bitmap, int16_t h, int16_t w) {
    for (int16_t y = 0; y < h; y++) {
        for (int16_t x = 0; x < w; x++) {
            // Only process occupied tiles (value == 1)
            if (bitmap[y * w + x] != 1) {
                continue;
            }

            // Check all 4 directions
            for (int16_t i = 0; i < 4; i++) {
                int16_t yy = y + DIRECTIONS_Y[i];
                int16_t xx = x + DIRECTIONS_X[i];

                // If neighbor is within bounds and empty, mark as padding (value == 2)
                if (0 <= yy && yy < h && 0 <= xx && xx < w && bitmap[yy * w + xx] == 0) {
                    bitmap[yy * w + xx] = 2;
                }
            }
        }
    }
}

/**
 * @brief Group connected tiles in a binary bitmap into polyominoes
 *
 * This is the main entry point for polyomino extraction. It processes a binary
 * bitmap to identify connected components (polyominoes), optionally applies padding,
 * normalizes coordinates, and returns a sorted array of polyominoes.
 *
 * Algorithm steps:
 * 1. Optionally add padding around occupied tiles based on mode
 * 2. Initialize group IDs for each occupied position
 * 3. For each unvisited occupied tile, perform flood-fill to find connected component
 * 4. Normalize coordinates to be relative to polyomino's bounding box origin
 * 5. Sort polyominoes by size (largest first) for optimal packing
 *
 * @param bitmap_input Flattened 2D array (height*width) of uint8_t values
 * @param width Width of the bitmap in tiles
 * @param height Height of the bitmap in tiles
 * @param mode Padding mode:
 *             - 0: No padding (strict connectivity)
 *             - 1: Disconnected padding (pad isolated tiles differently)
 *             - 2: Connected padding (pad all occupied tiles)
 *
 * @return Pointer to a newly allocated PolyominoArray, sorted by polyomino size (descending)
 *
 * @note The bitmap_input is modified during processing
 * @note Returned polyominoes have normalized coordinates (relative to their bounding box)
 * @note Caller must free the result using free_polyomino_array()
 */
PolyominoArray * group_tiles(
    uint8_t *bitmap_input,
    int16_t width,
    int16_t height,
    int8_t mode
) {
    // Allocate and initialize polyomino array
    PolyominoArray *polyomino_array = (PolyominoArray *)malloc(sizeof(PolyominoArray));
    PolyominoArray_init(polyomino_array, 16);

    // Add padding if mode is not 0
    if (mode != 0) {
        add_padding(bitmap_input, height, width);
    }

    // Create groups array with unique IDs
    int16_t *groups = (int16_t *)calloc((size_t)(height * width), sizeof(int16_t));
    CHECK_ALLOC(groups, "failed to allocate groups array for connected component labeling");

    // Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for (int16_t i = 0; i < height * width; i++) {
        if (bitmap_input[i]) {
            groups[i] = i + 1;
        }
    }

    // Process each cell to find connected components
    for (int16_t y = 0; y < height; y++) {
        for (int16_t x = 0; x < width; x++) {
            int16_t group_id = groups[y * width + x];
            // Check if this position has a valid group and the corresponding bitmap cell is non-zero
            // group_id = y * width + x + 1, so group_id - 1 is the flat index
            if (group_id == 0 || bitmap_input[group_id - 1] == 0) {
                continue;
            }

            // Find connected tiles - returns CoordinateArray
            CoordinateArray connected_tiles = find_connected_tiles(groups, bitmap_input,
                                                                    height, width, y, x, mode);
            if (connected_tiles.size == 0) {
                // Clean up empty CoordinateArray
                CoordinateArray_cleanup(&connected_tiles);
                continue;
            }

            // Find bounding box directly from CoordinateArray data
            // Initialize with first coordinate
            int16_t min_y = connected_tiles.data[0].y;
            int16_t min_x = connected_tiles.data[0].x;

            // Find min coordinates through all coordinates
            for (int16_t i = 1; i < connected_tiles.size; i++) {
                int16_t tile_y = connected_tiles.data[i].y;
                int16_t tile_x = connected_tiles.data[i].x;

                if (tile_y < min_y) {
                    min_y = tile_y;
                }

                if (tile_x < min_x) {
                    min_x = tile_x;
                }
            }

            // Normalize coordinates by subtracting min_y and min_x
            for (int16_t i = 0; i < connected_tiles.size; i++) {
                connected_tiles.data[i].y -= min_y;
                connected_tiles.data[i].x -= min_x;
            }

            // Create polyomino with normalized mask and offsets
            Polyomino polyomino = {.mask = connected_tiles,
                                   .offset_y = min_y,
                                   .offset_x = min_x};
            PolyominoArray_push(polyomino_array, polyomino);

            // Mark this position as processed
            bitmap_input[y * width + x] = 0;
        }
    }

    // Sort polyominoes by mask length (descending order) before returning
    qsort((void *)polyomino_array->data,
          (size_t)polyomino_array->size,
          sizeof(Polyomino),
          compare_polyomino_by_mask_length);

    free((void *)groups);
    return polyomino_array;
}

/**
 * @brief Free a polyomino array allocated by group_tiles
 *
 * Deallocates all memory associated with a PolyominoArray including the array
 * structure itself and all contained polyominoes. This is the proper cleanup
 * function for arrays returned by group_tiles().
 *
 * @param polyomino_array Pointer to the PolyominoArray to free
 * @return The number of polyominoes that were freed
 *
 * @note Passing NULL will trigger an assertion error if error checking is enabled
 * @note After calling this function, the pointer is invalid
 */
int free_polyomino_array(PolyominoArray *polyomino_array) {
    CHECK_NULL(polyomino_array, "polyomino_array pointer is NULL");

    int num_polyominoes = polyomino_array->size;
    PolyominoArray_cleanup(polyomino_array);
    free((void *)polyomino_array);
    return num_polyominoes;
}

