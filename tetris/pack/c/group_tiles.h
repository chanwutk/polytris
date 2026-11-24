/**
 * @file group_tiles.h
 * @brief Tile grouping and polyomino extraction from binary bitmaps
 *
 * This header provides the main API for converting a binary bitmap (where 1
 * represents an occupied tile and 0 represents empty space) into a collection
 * of polyominoes using flood-fill connected component analysis.
 *
 * The grouping algorithm supports multiple padding modes to handle different
 * use cases (isolated tiles, connected regions, etc.).
 */

#ifndef GROUP_TILES_H
#define GROUP_TILES_H

#include <stdint.h>
#include "utilities.h"

/**
 * @brief Group connected tiles in a binary bitmap into polyominoes
 *
 * This function performs flood-fill connected component analysis on a binary
 * bitmap to identify and extract all connected regions (polyominoes). Each
 * polyomino is represented as a coordinate array plus offset information.
 *
 * The bitmap is treated as a flattened 2D array in row-major order.
 *
 * @param bitmap_input Flattened 2D array (height*width) of uint8_t values,
 *                     where 1 indicates an occupied tile and 0 indicates empty space
 * @param width Width of the bitmap in tiles
 * @param height Height of the bitmap in tiles
 * @param mode Padding mode to apply during grouping:
 *             - 0: No padding (only group strictly connected tiles)
 *             - 1: Disconnected padding (treat isolated tiles separately)
 *             - 2: Connected padding (add padding around connected regions)
 *
 * @return Pointer to a newly allocated PolyominoArray containing all discovered
 *         polyominoes, or NULL on allocation failure. Caller must free using
 *         free_polyomino_array().
 *
 * @note The input bitmap is not modified
 * @note Connectivity is 4-connected (orthogonal neighbors only)
 * @warning Caller is responsible for freeing the returned PolyominoArray
 */
PolyominoArray * group_tiles(
    uint8_t *bitmap_input,
    int16_t width,
    int16_t height,
    int8_t mode
);

/**
 * @brief Free a polyomino array allocated by group_tiles
 *
 * Deallocates all memory associated with a PolyominoArray, including the
 * array structure itself and all contained polyominoes. After calling this
 * function, the pointer is invalid and should not be used.
 *
 * @param polyomino_array Pointer to the PolyominoArray to free (allocated by group_tiles)
 * @return The number of polyominoes that were freed
 *
 * @note This function handles nested cleanup automatically
 * @note Passing NULL is safe (returns 0)
 */
int free_polyomino_array(PolyominoArray *polyomino_array);

#endif // GROUP_TILES_H
