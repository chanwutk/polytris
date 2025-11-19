/**
 * @file utilities.h
 * @brief Core data structures and utilities for the packing library
 *
 * This header defines all the fundamental data structures used throughout the
 * packing algorithms, including coordinates, polyominoes, dynamic arrays, and
 * collage representations. It also declares the API functions for manipulating
 * these structures.
 *
 * @note All dynamic arrays follow a consistent pattern: init, push, cleanup
 * @note Memory management is the caller's responsibility for nested structures
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

// ============================================================================
// Structure Definitions
// ============================================================================

/**
 * @brief Represents a 2D coordinate or point in a grid
 *
 * This structure stores a single position in a 2D coordinate system using
 * 16-bit signed integers. Used to represent tile positions, offsets, and
 * polyomino shape components.
 */
typedef struct Coordinate {
    int16_t y;  /**< Y-coordinate (row) in the grid */
    int16_t x;  /**< X-coordinate (column) in the grid */
} Coordinate;

/**
 * @brief Dynamic array of Coordinate structures
 *
 * A resizable array that stores coordinate points. Used to represent polyomino
 * shapes, tile positions, and regions. Supports standard dynamic array operations
 * (init, push, cleanup).
 */
typedef struct CoordinateArray {
    Coordinate *data;  /**< Pointer to dynamically allocated array of coordinates */
    int size;          /**< Current number of coordinates in the array */
    int capacity;      /**< Allocated capacity (total slots available) */
} CoordinateArray;

/**
 * @brief Polyomino shape with coordinate-based mask and origin offset
 *
 * A polyomino is a connected set of tiles represented as a list of relative
 * coordinates (mask) plus an offset indicating the polyomino's position in
 * the original frame or grid.
 */
typedef struct Polyomino {
    CoordinateArray mask;  /**< Shape represented as array of relative coordinates */
    int16_t offset_y;      /**< Y-offset of polyomino origin in source grid */
    int16_t offset_x;      /**< X-offset of polyomino origin in source grid */
} Polyomino;

/**
 * @brief Dynamic array of Polyomino structures
 *
 * A resizable array that stores polyominoes. Used during tile grouping and
 * packing operations. Supports standard dynamic array operations.
 */
typedef struct PolyominoArray {
    Polyomino *data;  /**< Pointer to dynamically allocated array of polyominoes */
    int size;         /**< Current number of polyominoes in the array */
    int capacity;     /**< Allocated capacity (total slots available) */
} PolyominoArray;

/**
 * @brief Represents a placement position in a 2D grid
 *
 * Simple structure to store where a polyomino should be placed in the collage.
 * Used as a return value for placement algorithms.
 */
typedef struct Placement {
    int16_t y;  /**< Y-position (row) in the collage */
    int16_t x;  /**< X-position (column) in the collage */
} Placement;

/**
 * @brief Complete position information for a polyomino in a packed collage
 *
 * This structure stores both the original position of a polyomino in its source
 * frame and its packed position in the final collage. Used as the primary output
 * of packing algorithms to enable reconstruction and visualization.
 */
typedef struct PolyominoPosition {
    int16_t oy;              /**< Original Y-offset in the source video frame */
    int16_t ox;              /**< Original X-offset in the source video frame */
    int16_t py;              /**< Packed Y-position in the collage */
    int16_t px;              /**< Packed X-position in the collage */
    int32_t frame;           /**< Source frame index (which video frame this came from) */
    CoordinateArray shape;   /**< Shape of the polyomino as coordinate array */
} PolyominoPosition;

/**
 * @brief Dynamic array of PolyominoPosition structures
 *
 * A resizable array that stores polyomino positions. Each element represents
 * one packed polyomino. A single collage is typically represented by one
 * PolyominoPositionArray.
 */
typedef struct PolyominoPositionArray {
    PolyominoPosition *data;  /**< Pointer to dynamically allocated array of positions */
    int size;                 /**< Current number of positions in the array */
    int capacity;             /**< Allocated capacity (total slots available) */
} PolyominoPositionArray;

/**
 * @brief Collection of multiple collages
 *
 * Stores an array of collages, where each collage is a PolyominoPositionArray.
 * Used when packing produces multiple output images (e.g., when tiles don't
 * all fit in a single collage).
 */
typedef struct CollageArray {
    PolyominoPositionArray *data;  /**< Pointer to array of collages */
    int size;                      /**< Current number of collages */
    int capacity;                  /**< Allocated capacity for collages */
} CollageArray;

/**
 * @brief Dynamic array of uint8_t pointers
 *
 * A resizable array of pointers to uint8_t arrays. Primarily used for managing
 * pools of occupied tile bitmaps for multiple collages. The cleanup function
 * performs two-level deallocation (frees pointed-to arrays, then the pointer array).
 */
typedef struct U8PArray {
    uint8_t **data;  /**< Array of pointers to uint8_t arrays */
    int size;        /**< Current number of pointer elements */
    int capacity;    /**< Allocated capacity for pointers */
} U8PArray;

/**
 * @brief Dynamic array of integers
 *
 * A resizable array of int values. Used for tracking sizes of empty spaces,
 * region areas, and other integer metadata during packing operations.
 */
typedef struct IntArray {
    int *data;      /**< Pointer to dynamically allocated array of integers */
    int size;       /**< Current number of integers in the array */
    int capacity;   /**< Allocated capacity (total slots available) */
} IntArray;

/**
 * @brief Polyomino data bundled with frame index and size for sorting
 *
 * An extended polyomino structure that includes the source frame index and
 * precomputed size. Used during preprocessing and sorting operations before
 * packing algorithms run. Allows efficient sorting by size without recomputing.
 */
typedef struct PolyominoWithFrame {
    CoordinateArray shape;  /**< Shape of the polyomino as coordinate array */
    int oy;                 /**< Original Y-offset in the source frame */
    int ox;                 /**< Original X-offset in the source frame */
    int frame;              /**< Source frame index */
    int size;               /**< Precomputed size (number of tiles in polyomino) */
} PolyominoWithFrame;

/**
 * @brief Dynamic array of PolyominoWithFrame structures
 *
 * A resizable array that stores polyominoes with their frame and size information.
 * Used during the sorting and preprocessing stages of packing algorithms.
 */
typedef struct PolyominoWithFrameArray {
    PolyominoWithFrame *data;  /**< Pointer to dynamically allocated array */
    int size;                  /**< Current number of elements in the array */
    int capacity;              /**< Allocated capacity (total slots available) */
} PolyominoWithFrameArray;

// ============================================================================
// Function Declarations - Polyomino and PolyominoArray
// ============================================================================

/**
 * @brief Free the memory used by a polyomino's mask array
 *
 * Deallocates the coordinate array (mask) contained within the polyomino structure.
 * Does not free the polyomino structure itself, only its internal mask data.
 *
 * @param polyomino Pointer to the polyomino to clean up
 * @note The polyomino pointer itself remains valid but its mask is freed
 */
void Polyomino_cleanup(Polyomino *polyomino);

/**
 * @brief Initialize a polyomino array with a given initial capacity
 *
 * Allocates memory for a PolyominoArray and sets its initial capacity.
 * The array starts empty (size = 0) but with preallocated space.
 *
 * @param array Pointer to the PolyominoArray structure to initialize
 * @param initial_capacity Initial number of slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoArray_init(PolyominoArray *array, int initial_capacity);

/**
 * @brief Add a polyomino to the array, expanding capacity if necessary
 *
 * Appends a polyomino to the end of the array. If the array is full, it
 * automatically reallocates with increased capacity (typically doubles).
 *
 * @param array Pointer to the PolyominoArray to modify
 * @param value The Polyomino value to append (copied into the array)
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoArray_push(PolyominoArray *array, Polyomino value);

/**
 * @brief Free the polyomino array and all contained polyominoes
 *
 * Performs deep cleanup: first calls Polyomino_cleanup on each contained
 * polyomino, then frees the array's data pointer. After this call, the
 * array must be reinitialized before reuse.
 *
 * @param array Pointer to the PolyominoArray to clean up
 * @note This performs nested cleanup on all polyominoes in the array
 */
void PolyominoArray_cleanup(PolyominoArray *array);

// ============================================================================
// Function Declarations - CoordinateArray
// ============================================================================

/**
 * @brief Initialize a coordinate array with a given initial capacity
 *
 * Allocates memory for a CoordinateArray and sets its initial capacity.
 * The array starts empty (size = 0) but with preallocated space.
 *
 * @param arr Pointer to the CoordinateArray structure to initialize
 * @param initial_capacity Initial number of coordinate slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int CoordinateArray_init(CoordinateArray *arr, int initial_capacity);

/**
 * @brief Add a coordinate to the array, expanding capacity if necessary
 *
 * Appends a coordinate to the end of the array. If the array is full, it
 * automatically reallocates with increased capacity (typically doubles).
 *
 * @param arr Pointer to the CoordinateArray to modify
 * @param coord The Coordinate value to append (copied into the array)
 * @return 0 on success, non-zero on allocation failure
 */
int CoordinateArray_push(CoordinateArray *arr, Coordinate coord);

/**
 * @brief Free the memory used by a coordinate array
 *
 * Deallocates the data pointer of the coordinate array. After this call,
 * the array must be reinitialized before reuse.
 *
 * @param arr Pointer to the CoordinateArray to clean up
 */
void CoordinateArray_cleanup(CoordinateArray *arr);

// ============================================================================
// Function Declarations - PolyominoPositionArray
// ============================================================================

/**
 * @brief Initialize a PolyominoPositionArray with a given initial capacity
 *
 * @param arr Pointer to the PolyominoPositionArray structure to initialize
 * @param initial_capacity Initial number of position slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoPositionArray_init(PolyominoPositionArray *arr, int initial_capacity);

/**
 * @brief Add a PolyominoPosition to the array, expanding capacity if necessary
 *
 * @param arr Pointer to the PolyominoPositionArray to modify
 * @param pos The PolyominoPosition value to append (copied into the array)
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoPositionArray_push(PolyominoPositionArray *arr, PolyominoPosition pos);

/**
 * @brief Free the PolyominoPositionArray and all contained positions
 *
 * Performs deep cleanup: frees the shape CoordinateArray for each position,
 * then frees the array's data pointer.
 *
 * @param arr Pointer to the PolyominoPositionArray to clean up
 * @note This performs nested cleanup on all positions in the array
 */
void PolyominoPositionArray_cleanup(PolyominoPositionArray *arr);

// ============================================================================
// Function Declarations - CollageArray
// ============================================================================

/**
 * @brief Initialize a CollageArray with a given initial capacity
 *
 * @param list Pointer to the CollageArray structure to initialize
 * @param initial_capacity Initial number of collage slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int CollageArray_init(CollageArray *list, int initial_capacity);

/**
 * @brief Add a PolyominoPositionArray (collage) to the array
 *
 * Appends an entire collage to the array of collages. The collage is copied
 * into the array.
 *
 * @param list Pointer to the CollageArray to modify
 * @param arr The PolyominoPositionArray (collage) to append
 * @return 0 on success, non-zero on allocation failure
 */
int CollageArray_push(CollageArray *list, PolyominoPositionArray arr);

/**
 * @brief Free the CollageArray and all contained collages
 *
 * Performs deep cleanup: calls PolyominoPositionArray_cleanup on each collage,
 * then frees the array's data pointer.
 *
 * @param list Pointer to the CollageArray to clean up
 * @note This performs nested cleanup on all collages in the array
 */
void CollageArray_cleanup(CollageArray *list);

// ============================================================================
// Function Declarations - U8PArray
// ============================================================================

/**
 * @brief Initialize a U8PArray with a given initial capacity
 *
 * @param arr Pointer to the U8PArray structure to initialize
 * @param initial_capacity Initial number of pointer slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int U8PArray_init(U8PArray *arr, int initial_capacity);

/**
 * @brief Add a uint8_t pointer to the array, expanding capacity if necessary
 *
 * @param arr Pointer to the U8PArray to modify
 * @param value The uint8_t pointer to append (pointer is stored, not copied)
 * @return 0 on success, non-zero on allocation failure
 * @note The array stores the pointer itself, not a copy of the data
 */
int U8PArray_push(U8PArray *arr, uint8_t *value);

/**
 * @brief Free the U8PArray and all pointed-to arrays
 *
 * Performs two-level cleanup: first frees each uint8_t array pointed to by
 * the stored pointers, then frees the pointer array itself.
 *
 * @param arr Pointer to the U8PArray to clean up
 * @note This performs nested deallocation of all pointed-to arrays
 */
void U8PArray_cleanup(U8PArray *arr);

// ============================================================================
// Function Declarations - IntArray
// ============================================================================

/**
 * @brief Initialize an IntArray with a given initial capacity
 *
 * @param arr Pointer to the IntArray structure to initialize
 * @param initial_capacity Initial number of integer slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int IntArray_init(IntArray *arr, int initial_capacity);

/**
 * @brief Add an integer to the array, expanding capacity if necessary
 *
 * @param arr Pointer to the IntArray to modify
 * @param value The integer value to append
 * @return 0 on success, non-zero on allocation failure
 */
int IntArray_push(IntArray *arr, int value);

/**
 * @brief Free the memory used by an IntArray
 *
 * Deallocates the data pointer of the integer array.
 *
 * @param arr Pointer to the IntArray to clean up
 */
void IntArray_cleanup(IntArray *arr);

// ============================================================================
// Function Declarations - PolyominoWithFrameArray
// ============================================================================

/**
 * @brief Initialize a PolyominoWithFrameArray with a given initial capacity
 *
 * @param arr Pointer to the PolyominoWithFrameArray structure to initialize
 * @param initial_capacity Initial number of element slots to allocate
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoWithFrameArray_init(PolyominoWithFrameArray *arr, int initial_capacity);

/**
 * @brief Add a PolyominoWithFrame to the array, expanding capacity if necessary
 *
 * @param arr Pointer to the PolyominoWithFrameArray to modify
 * @param item The PolyominoWithFrame value to append (copied into the array)
 * @return 0 on success, non-zero on allocation failure
 */
int PolyominoWithFrameArray_push(PolyominoWithFrameArray *arr, PolyominoWithFrame item);

/**
 * @brief Free the PolyominoWithFrameArray and all contained items
 *
 * Performs deep cleanup: frees the shape CoordinateArray for each item,
 * then frees the array's data pointer.
 *
 * @param arr Pointer to the PolyominoWithFrameArray to clean up
 * @note This performs nested cleanup on all items in the array
 */
void PolyominoWithFrameArray_cleanup(PolyominoWithFrameArray *arr);

#endif // UTILITIES_H

