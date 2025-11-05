#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Error Handling Configuration
// ============================================================================
// Define DISABLE_ERROR_CHECKS to completely remove all error checking at compile time
// Usage: compile with -DDISABLE_ERROR_CHECKS to disable all checks
// When disabled, error checks have zero runtime overhead

// Exit codes for different error types
#define EXIT_ERROR_NULL_POINTER 1
#define EXIT_ERROR_ALLOCATION_FAILED 2
#define EXIT_ERROR_ASSERTION_FAILED 3

// ============================================================================
// Core Error Checking Macros
// ============================================================================

#ifndef DISABLE_ERROR_CHECKS

// Check if a pointer is NULL, exit with error if true
#define CHECK_NULL(ptr, error_msg) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: NULL pointer - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_NULL_POINTER); \
        } \
    } while (0)

// Check if a condition is true, exit with error if false
#define ASSERT(cond, error_msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: Condition failed - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_ASSERTION_FAILED); \
        } \
    } while (0)

// Check if malloc/realloc succeeded, exit with error if failed
#define CHECK_ALLOC(ptr, error_msg) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "ERROR [%s:%d in %s]: Memory allocation failed - %s\n", \
                    __FILE__, __LINE__, __func__, (error_msg)); \
            exit(EXIT_ERROR_ALLOCATION_FAILED); \
        } \
    } while (0)

#else  // DISABLE_ERROR_CHECKS is defined

// When error checking is disabled, all macros become no-ops
#define CHECK_NULL(ptr, error_msg) ((void)0)
#define ASSERT(cond, error_msg) ((void)0)
#define CHECK_ALLOC(ptr, error_msg) ((void)0)

#endif  // DISABLE_ERROR_CHECKS

#endif  // ERROR_HANDLING_H
