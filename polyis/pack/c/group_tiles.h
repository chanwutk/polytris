#ifndef GROUP_TILES_H
#define GROUP_TILES_H

#include <stdint.h>
#include "utilities.h"

// Main function to group tiles into polyominoes
// bitmap_input: 2D array (flattened) of uint8_t representing the grid of tiles
//               where 1 indicates a tile with detection and 0 indicates no detection
// width: width of the bitmap
// height: height of the bitmap
// mode: The mode of tile padding to apply
//          - 0: No padding
//          - 1: Disconnected padding
//          - 2: Connected padding
// Returns: Pointer to PolyominoArray containing all found polyominoes
PolyominoArray * group_tiles(
    uint8_t *bitmap_input,
    int16_t width,
    int16_t height,
    int8_t mode
);

// Free a polyomino array allocated by group_tiles
// Returns the number of polyominoes that were freed
int free_polyomino_array(PolyominoArray *polyomino_array);

#endif // GROUP_TILES_H
