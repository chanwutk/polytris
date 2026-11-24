/**
 * @file errors.h
 * @brief Error handling and assertion macros for the packing library
 *
 * This header provides compile-time configurable error checking macros that can be
 * completely disabled for production builds. When enabled, these macros provide
 * detailed error messages with file, line, and function information.
 *
 * @note Define DISABLE_ERROR_CHECKS at compile time to remove all error checking overhead
 * @note Compile with -DDISABLE_ERROR_CHECKS to disable all runtime checks
 */

#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Error Handling Configuration
// ============================================================================

/**
 * @def DISABLE_ERROR_CHECKS
 * @brief Compile-time flag to disable all error checking
 *
 * When defined, all error checking macros become no-ops, resulting in zero
 * runtime overhead. Useful for production builds after thorough testing.
 */

/**
 * @def EXIT_ERROR_NULL_POINTER
 * @brief Exit code for NULL pointer errors
 */
#define EXIT_ERROR_NULL_POINTER 1

/**
 * @def EXIT_ERROR_ALLOCATION_FAILED
 * @brief Exit code for memory allocation failures
 */
#define EXIT_ERROR_ALLOCATION_FAILED 2

/**
 * @def EXIT_ERROR_ASSERTION_FAILED
 * @brief Exit code for assertion failures
 */
#define EXIT_ERROR_ASSERTION_FAILED 3

// ============================================================================
// Core Error Checking Macros
// ============================================================================

#ifndef DISABLE_ERROR_CHECKS

/**
 * @def CHECK_NULL(ptr, error_msg)
 * @brief Check if a pointer is NULL and exit with an error if true
 *
 * This macro validates that the given pointer is not NULL. If the pointer is NULL,
 * it prints an error message including file location, line number, and function name,
 * then exits the program with EXIT_ERROR_NULL_POINTER.
 *
 * @param ptr The pointer to check for NULL
 * @param error_msg A descriptive error message to display if the check fails
 *
 * @note This macro is a do-while(0) block, so it can be safely used in all contexts
 * @note When DISABLE_ERROR_CHECKS is defined, this becomes a no-op
 *
 * Example usage:
 * @code
 * int *data = malloc(sizeof(int) * 100);
 * CHECK_NULL(data, "Failed to allocate data array");
 * @endcode
 */
#define CHECK_NULL(ptr, error_msg) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: NULL pointer - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_NULL_POINTER); \
        } \
    } while (0)

/**
 * @def ASSERT(cond, error_msg)
 * @brief Assert that a condition is true, exit with an error if false
 *
 * This macro validates that the given condition evaluates to true. If the condition
 * is false, it prints an error message including file location, line number, and
 * function name, then exits the program with EXIT_ERROR_ASSERTION_FAILED.
 *
 * @param cond The condition to evaluate (must be true)
 * @param error_msg A descriptive error message to display if the assertion fails
 *
 * @note This macro is a do-while(0) block, so it can be safely used in all contexts
 * @note When DISABLE_ERROR_CHECKS is defined, this becomes a no-op
 *
 * Example usage:
 * @code
 * ASSERT(width > 0 && height > 0, "Width and height must be positive");
 * @endcode
 */
#define ASSERT(cond, error_msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: Condition failed - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_ASSERTION_FAILED); \
        } \
    } while (0)

/**
 * @def CHECK_ALLOC(ptr, error_msg)
 * @brief Check if a memory allocation succeeded, exit with an error if it failed
 *
 * This macro validates that a malloc/calloc/realloc operation succeeded by checking
 * if the returned pointer is not NULL. If allocation failed, it prints an error
 * message including file location, line number, and function name, then exits the
 * program with EXIT_ERROR_ALLOCATION_FAILED.
 *
 * @param ptr The pointer returned by malloc/calloc/realloc to check
 * @param error_msg A descriptive error message to display if allocation failed
 *
 * @note This macro is a do-while(0) block, so it can be safely used in all contexts
 * @note When DISABLE_ERROR_CHECKS is defined, this becomes a no-op
 * @warning Using this macro does not free any previously allocated memory on failure
 *
 * Example usage:
 * @code
 * int *buffer = malloc(sizeof(int) * size);
 * CHECK_ALLOC(buffer, "Failed to allocate buffer");
 * @endcode
 */
#define CHECK_ALLOC(ptr, error_msg) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: Memory allocation failed - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_ALLOCATION_FAILED); \
        } \
    } while (0)

#else  // DISABLE_ERROR_CHECKS is defined

/**
 * @brief No-op versions of error checking macros when DISABLE_ERROR_CHECKS is defined
 *
 * When error checking is disabled at compile time, all checking macros become
 * no-ops with zero runtime overhead.
 */
#define CHECK_NULL(ptr, error_msg) ((void)0)
#define ASSERT(cond, error_msg) ((void)0)
#define CHECK_ALLOC(ptr, error_msg) ((void)0)

#endif  // DISABLE_ERROR_CHECKS

#endif  // ERROR_HANDLING_H
