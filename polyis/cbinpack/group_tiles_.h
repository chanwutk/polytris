#ifndef GROUP_TILES_H
#define GROUP_TILES_H

#include "utilities.h"

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
);

// Free a polyomino stack allocated by group_tiles
// Returns the number of polyominoes that were freed
int free_polyomino_stack_(PolyominoStack *polyomino_stack);

#endif // GROUP_TILES_H
