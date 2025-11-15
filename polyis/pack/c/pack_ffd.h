/**
 * @file pack_ffd.h
 * @brief First-Fit-Descending (FFD) polyomino packing algorithm
 *
 * This header provides the main API for packing multiple polyominoes into
 * fixed-size collages using a First-Fit-Descending strategy. The algorithm
 * sorts polyominoes by size (largest first) and places them into collages
 * using various fit strategies.
 *
 * @note This implementation supports multiple packing modes for different use cases
 */

#ifndef PACK_FFD_H
#define PACK_FFD_H

#include "utilities.h"

/**
 * @brief Packing strategy modes for collage selection
 *
 * Defines how the packing algorithm selects which collage to place a
 * polyomino into when multiple collages could accommodate it.
 */
typedef enum PackMode {
    Easiest_Fit,  /**< Pack into collage with most empty space (minimizes fragmentation) */
    First_Fit,    /**< Pack into first collage that fits (fastest, may waste space) */
    Best_Fit      /**< Pack into collage with least empty space that fits (tightest packing) */
} PackMode;

/**
 * @brief Pack multiple polyomino arrays into fixed-size collages
 *
 * This function implements a First-Fit-Descending packing algorithm that takes
 * multiple arrays of polyominoes and packs them into a collection of fixed-size
 * collages. Polyominoes are sorted by size (largest first) before packing to
 * improve packing efficiency.
 *
 * @param polyominoes_arrays Array of pointers to PolyominoArray structures,
 *                           each containing polyominoes from a single frame or group
 * @param num_arrays Number of PolyominoArray pointers in the array
 * @param h Height of each collage in tiles
 * @param w Width of each collage in tiles
 * @param mode Packing mode that determines collage selection strategy
 *             (Easiest_Fit, First_Fit, or Best_Fit)
 *
 * @return Pointer to a newly allocated CollageArray containing all packed collages,
 *         or NULL on allocation failure. Caller must free using CollageArray_cleanup().
 *
 * @note Polyominoes are copied into the result, originals are not modified
 * @note New collages are created as needed when existing ones fill up
 * @note The algorithm may leave some collages partially filled
 * @warning Caller is responsible for freeing the returned CollageArray
 */
CollageArray* pack_all_(PolyominoArray **polyominoes_arrays, int num_arrays, int h, int w, PackMode mode);

#endif // PACK_FFD_H
