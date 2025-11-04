#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "utilities.h"

// Direction arrays for 4-connectivity (up, left, down, right)
static const char DIRECTIONS_I[4] = {-1, 0, 1, 0};
static const char DIRECTIONS_J[4] = {0, -1, 0, 1};

// Comparison function for qsort to sort polyominoes by mask length (descending order)
// Returns negative if a should come before b, positive if b should come before a
static int compare_polyomino_by_mask_length(const void *a, const void *b) {
    const Polyomino *poly_a = (const Polyomino *)a;
    const Polyomino *poly_b = (const Polyomino *)b;
    // Compare by mask length (top field of IntStack) in descending order
    // Larger masks first (negative return means a comes before b)
    return poly_b->mask.top - poly_a->mask.top;
}

// Helper function to find connected tiles using flood fill algorithm
// This function modifies the bitmap in-place to mark visited tiles
static IntStack _find_connected_tiles(
    unsigned int *bitmap,
    unsigned short h,
    unsigned short w,
    unsigned short start_i,
    unsigned short start_j,
    unsigned char *bitmap_input,
    int mode
) {
    IntStack filled, stack;
    unsigned short i, j, _i, _j;
    int di;
    unsigned int value = bitmap[start_i * w + start_j];
    unsigned char curr_occupancy;
    unsigned int next_group;
    
    // Initialize stacks
    if (IntStack_init(&filled, 16) != 0) {
        // Return empty IntStack on initialization failure
        IntStack_cleanup(&filled);
        return filled;
    }
    
    if (IntStack_init(&stack, 16) != 0) {
        // Initialization failed, cleanup filled and return empty
        IntStack_cleanup(&filled);
        return filled;
    }

    // Push initial coordinates
    IntStack_push(&stack, start_i);
    IntStack_push(&stack, start_j);

    // Flood fill algorithm
    while (stack.top > 0) {
        // Pop coordinates from stack
        j = stack.data[stack.top - 1];
        i = stack.data[stack.top - 2];
        stack.top -= 2;

        // Mark current position as visited and add to result
        bitmap[i * w + j] = value;
        IntStack_push(&filled, i);
        IntStack_push(&filled, j);

        curr_occupancy = bitmap_input[i * w + j];
        
        // Check all 4 directions for unvisited connected tiles
        for (di = 0; di < 4; di++) {
            _i = i + DIRECTIONS_I[di];
            _j = j + DIRECTIONS_J[di];
            
            // Check bounds
            if (0 <= _i && _i < h && 0 <= _j && _j < w) {
                next_group = bitmap[_i * w + _j];
                
                // Add neighbors that are non-zero and different from current value
                // (meaning they haven't been visited yet)
                if (next_group != 0 && next_group != value) {
                    // Check padding mode conditions
                    if (mode == 0 || mode == 2 || 
                        (mode == 1 && (curr_occupancy == 1 || bitmap_input[_i * w + _j] == 1))) {
                        IntStack_push(&stack, _i);
                        IntStack_push(&stack, _j);
                    }
                }
            }
        }
    }

    // Free the stack's data before returning
    IntStack_cleanup(&stack);
    return filled;
}

// Add padding to bitmap based on tilepadding_mode
// mode 1: Connected padding - pad neighbors of occupied tiles
// mode 2: Disconnected padding - pad all neighbors
static void _add_padding(
    unsigned char *bitmap,
    unsigned short h,
    unsigned short w
) {
    int i, j, di;
    char _i, _j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // Only process occupied tiles (value == 1)
            if (bitmap[i * w + j] != 1) {
                continue;
            }

            // Check all 4 directions
            for (di = 0; di < 4; di++) {
                _i = i + DIRECTIONS_I[di];
                _j = j + DIRECTIONS_J[di];

                // If neighbor is within bounds and empty, mark as padding (value == 2)
                if (0 <= _i && _i < h && 0 <= _j && _j < w && bitmap[_i * w + _j] == 0) {
                    bitmap[_i * w + _j] = 2;
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
// Returns: Pointer to PolyominoStack containing all found polyominoes
PolyominoStack * group_tiles_(
    unsigned char *bitmap_input,
    int width,
    int height,
    int tilepadding_mode
) {
    unsigned short h = (unsigned short)height;
    unsigned short w = (unsigned short)width;
    unsigned short group_id, min_i, min_j, tile_i, tile_j, num_pairs;
    int i, j, k;
    IntStack connected_tiles;
    Polyomino polyomino;
    PolyominoStack *polyomino_stack;
    unsigned int *groups;
    
    // Allocate and initialize polyomino stack
    polyomino_stack = (PolyominoStack *)malloc(sizeof(PolyominoStack));
    if (polyomino_stack == NULL) {
        return NULL;
    }
    PolyominoStack_init(polyomino_stack, 16);

    // Add padding if mode is not 0
    if (tilepadding_mode != 0) {
        _add_padding(bitmap_input, h, w);
    }

    // Create groups array with unique IDs
    groups = (unsigned int *)calloc((size_t)(h * w), sizeof(unsigned int));
    if (groups == NULL) {
        PolyominoStack_cleanup(polyomino_stack);
        free(polyomino_stack);
        return NULL;
    }
    
    // Mask groups by bitmap - only keep group IDs where bitmap has 1s
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            if (bitmap_input[i * w + j]) {
                groups[i * w + j] = i * w + j + 1;
            }
        }
    }

    // Process each cell to find connected components
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            group_id = groups[i * w + j];
            // Check if this position has a valid group and the corresponding bitmap cell is non-zero
            // group_id = i * w + j + 1, so group_id - 1 is the flat index
            if (group_id == 0 || bitmap_input[group_id - 1] == 0) {
                continue;
            }

            // Find connected tiles - returns IntStack
            connected_tiles = _find_connected_tiles(groups, h, w, (unsigned short)i, (unsigned short)j,
                                                    bitmap_input, tilepadding_mode);
            if (connected_tiles.top == 0) {
                // Clean up empty IntStack
                IntStack_cleanup(&connected_tiles);
                continue;
            }
            
            // Find bounding box directly from IntStack data
            num_pairs = (unsigned short)(connected_tiles.top / 2);
            
            // Initialize with first coordinate pair
            min_i = connected_tiles.data[0];
            min_j = connected_tiles.data[1];
            
            // Find min coordinates through all coordinate pairs
            for (k = 1; k < num_pairs; k++) {
                tile_i = connected_tiles.data[k * 2];
                tile_j = connected_tiles.data[(k * 2) + 1];
                
                if (tile_i < min_i) {
                    min_i = tile_i;
                }
                
                if (tile_j < min_j) {
                    min_j = tile_j;
                }
            }

            // Normalize coordinates by subtracting min_i and min_j
            for (k = 0; k < num_pairs; k++) {
                connected_tiles.data[k * 2] -= min_i;
                connected_tiles.data[(k * 2) + 1] -= min_j;
            }
            
            // Create polyomino with normalized mask and offsets
            polyomino.mask = connected_tiles;
            polyomino.offset_i = min_i;
            polyomino.offset_j = min_j;
            PolyominoStack_push(polyomino_stack, polyomino);

            // Mark this position as processed
            bitmap_input[i * w + j] = 0;
        }
    }

    // Sort polyominoes by mask length (descending order) before returning
    qsort((void *)polyomino_stack->data,
          (size_t)polyomino_stack->top,
          sizeof(Polyomino),
          compare_polyomino_by_mask_length);

    free((void *)groups);
    return polyomino_stack;
}

// Free a polyomino stack allocated by group_tiles
// Returns the number of polyominoes that were freed
int free_polyomino_stack_(PolyominoStack *polyomino_stack) {
    if (polyomino_stack == NULL) {
        return 0;
    }
    int num_polyominoes = polyomino_stack->top;
    PolyominoStack_cleanup(polyomino_stack);
    free((void *)polyomino_stack);
    return num_polyominoes;
}

