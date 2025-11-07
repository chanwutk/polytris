#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "utilities.h"
#include "errors.h"

// Direction arrays for 4-connectivity (up, left, down, right)
static const char DIRECTIONS_Y[4] = {-1, 0, 1, 0};
static const char DIRECTIONS_X[4] = {0, -1, 0, 1};

// Comparison function for qsort to sort polyominoes by mask length (descending order)
// Returns negative if a should come before b, positive if b should come before a
static int compare_polyomino_by_mask_length(const void *a, const void *b) {
    const Polyomino *poly_a = (const Polyomino *)a;
    const Polyomino *poly_b = (const Polyomino *)b;
    // Compare by mask length (size field of ShortArray) in descending order
    // Larger masks first (negative return means a comes before b)
    return poly_b->mask.size - poly_a->mask.size;
}

// Helper function to find connected tiles using flood fill algorithm
// This function modifies the bitmap in-place to mark visited tiles
static CoordinateArray _find_connected_tiles(
    short *bitmap,
    unsigned char *bitmap_input,
    short h,
    short w,
    short start_y,
    short start_x,
    char mode
) {
    CoordinateArray filled, stack;
    short y, x, yy, xx;
    int di;
    short value = bitmap[start_y * w + start_x];
    unsigned char curr_occupancy;
    short next_group;
    Coordinate coord;

    // Initialize arrays
    CoordinateArray_init(&filled, 16);
    CoordinateArray_init(&stack, 16);

    // Push initial coordinates
    coord.y = start_y;
    coord.x = start_x;
    CoordinateArray_push(&stack, coord);

    // Flood fill algorithm
    while (stack.size > 0) {
        // Pop coordinates from stack
        stack.size--;
        coord = stack.data[stack.size];
        x = stack.data[stack.size].x;
        y = stack.data[stack.size].y;

        if (bitmap[y * w + x] == value && (x != start_x || y != start_y)) {
            continue;  // Already visited
        }

        // Mark current position as visited and add to result
        bitmap[y * w + x] = value;
        coord.y = y;
        coord.x = x;
        CoordinateArray_push(&filled, coord);

        curr_occupancy = bitmap_input[y * w + x];

        // Check all 4 directions for unvisited connected tiles
        for (di = 0; di < 4; di++) {
            yy = y + DIRECTIONS_Y[di];
            xx = x + DIRECTIONS_X[di];

            // Check bounds
            if (0 <= yy && yy < h && 0 <= xx && xx < w) {
                next_group = bitmap[yy * w + xx];

                // Add neighbors that are non-zero and different from current value
                // (meaning they haven't been visited yet)
                if (next_group != 0 && next_group != value) {
                    // Check padding mode conditions
                    if (mode != 1 || curr_occupancy == 1 || bitmap_input[yy * w + xx] == 1) {
                        coord.y = yy;
                        coord.x = xx;
                        CoordinateArray_push(&stack, coord);
                    }
                }
            }
        }
    }

    // Free the array's data before returning
    CoordinateArray_cleanup(&stack);
    return filled;
}

// Add padding to bitmap based on tilepadding_mode
// mode 1: Connected padding - pad neighbors of occupied tiles
// mode 2: Disconnected padding - pad all neighbors
static void _add_padding(unsigned char *bitmap, short h, short w) {
    for (short y = 0; y < h; y++) {
        for (short x = 0; x < w; x++) {
            // Only process occupied tiles (value == 1)
            if (bitmap[y * w + x] != 1) {
                continue;
            }

            // Check all 4 directions
            for (short i = 0; i < 4; i++) {
                short yy = y + DIRECTIONS_Y[i];
                short xx = x + DIRECTIONS_X[i];

                // If neighbor is within bounds and empty, mark as padding (value == 2)
                if (0 <= yy && yy < h && 0 <= xx && xx < w && bitmap[yy * w + xx] == 0) {
                    bitmap[yy * w + xx] = 2;
                }
            }
        }
    }
}

// Main function to group tiles into polyominoes
// bitmap_input: 2D array (flattened) of uint8_t representing the grid of tiles
//               where 1 indicates a tile with detection and 0 indicates no detection
// width: width of the bitmap
// height: height of the bitmap
// tilepadding_mode: The mode of tile padding to apply
//                   - 0: No padding
//                   - 1: Connected padding
//                   - 2: Disconnected padding
// Returns: Pointer to PolyominoArray containing all found polyominoes
PolyominoArray * group_tiles_(
    unsigned char *bitmap_input,
    short width,
    short height,
    char tilepadding_mode
) {
    short group_id, min_i, min_j;
    int i, j, k, tile_i, tile_j;
    CoordinateArray connected_tiles;
    Polyomino polyomino;
    PolyominoArray *polyomino_array;
    short *groups;

    // Allocate and initialize polyomino array
    polyomino_array = (PolyominoArray *)malloc(sizeof(PolyominoArray));
    CHECK_NULL(polyomino_array, "failed to allocate PolyominoArray");
    PolyominoArray_init(polyomino_array, 16);

    // Add padding if mode is not 0
    if (tilepadding_mode != 0) {
        _add_padding(bitmap_input, height, width);
    }

    // Create groups array with unique IDs
    groups = (short *)calloc((size_t)(height * width), sizeof(short));
    CHECK_ALLOC(groups, "failed to allocate groups array for connected component labeling");

    // Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for (i = 0; i < height * width; i++) {
        if (bitmap_input[i]) {
            groups[i] = i + 1;
        }
    }

    // Process each cell to find connected components
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            group_id = groups[i * width + j];
            // Check if this position has a valid group and the corresponding bitmap cell is non-zero
            // group_id = i * w + j + 1, so group_id - 1 is the flat index
            if (group_id == 0 || bitmap_input[group_id - 1] == 0) {
                continue;
            }

            // Find connected tiles - returns CoordinateArray
            connected_tiles = _find_connected_tiles(groups, bitmap_input, height, width,
                                                    i, j, tilepadding_mode);
            if (connected_tiles.size == 0) {
                // Clean up empty CoordinateArray
                CoordinateArray_cleanup(&connected_tiles);
                continue;
            }

            // Find bounding box directly from CoordinateArray data
            // Initialize with first coordinate
            min_i = connected_tiles.data[0].y;
            min_j = connected_tiles.data[0].x;

            // Find min coordinates through all coordinates
            for (k = 1; k < connected_tiles.size; k++) {
                tile_i = connected_tiles.data[k].y;
                tile_j = connected_tiles.data[k].x;

                if (tile_i < min_i) {
                    min_i = tile_i;
                }

                if (tile_j < min_j) {
                    min_j = tile_j;
                }
            }

            // Normalize coordinates by subtracting min_i and min_j
            for (k = 0; k < connected_tiles.size; k++) {
                connected_tiles.data[k].y -= min_i;
                connected_tiles.data[k].x -= min_j;
            }

            // Create polyomino with normalized mask and offsets
            polyomino.mask = connected_tiles;
            polyomino.offset_i = min_i;
            polyomino.offset_j = min_j;
            PolyominoArray_push(polyomino_array, polyomino);

            // Mark this position as processed
            bitmap_input[i * width + j] = 0;
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

// Free a polyomino array allocated by group_tiles
// Returns the number of polyominoes that were freed
int free_polyomino_array_(PolyominoArray *polyomino_array) {
    CHECK_NULL(polyomino_array, "polyomino_array pointer is NULL");

    int num_polyominoes = polyomino_array->size;
    PolyominoArray_cleanup(polyomino_array);
    free((void *)polyomino_array);
    return num_polyominoes;
}

