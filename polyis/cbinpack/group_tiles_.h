#ifndef GROUP_TILES_H
#define GROUP_TILES_H

#include "utilities_.h"

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
    int width,
    int height,
    int tilepadding_mode
);

// Free a polyomino array allocated by group_tiles
// Returns the number of polyominoes that were freed
int free_polyomino_array_(PolyominoArray *polyomino_array);

#endif // GROUP_TILES_H
