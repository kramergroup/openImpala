#include <mpi.h>
#include <HYPRE.h>
#include <HYPRE_struct_ls.h> // For Struct interface
#include <HYPRE_struct_mv.h> // For Struct interface

#include <stdio.h>
#include <stdlib.h>

// Simple HYPRE error checking macro
#define HYPRE_CHECK_MSG(msg, ierr) do { \
    if (ierr != 0) { \
        char hypre_error_msg[256]; \
        HYPRE_DescribeError(ierr, hypre_error_msg); \
        fprintf(stderr, "FAIL: %s\n", msg); \
        fprintf(stderr, "  HYPRE Error: %s (Code: %d)\n", hypre_error_msg, ierr); \
        fflush(stderr); /* Ensure error message is printed */ \
        MPI_Abort(MPI_COMM_WORLD, ierr); \
    } else { \
        /* Optional: Print PASS message only if needed, commented out by default */ \
        /* printf("PASS: %s\n", msg); */ \
    } \
} while (0)

int main(int argc, char *argv[]) {
    int myid, num_procs;
    int ierr = 0;
    int final_exit_code = 0; // Use to track overall success/failure

    // Initialize MPI
    printf("Calling MPI_Init...\n"); fflush(stdout);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (myid == 0) {
        printf("Running HYPRE Standalone Test (hypre_test.cpp)...\n");
        printf("MPI Procs: %d\n", num_procs);
        if (num_procs > 1) {
            printf("WARNING: This test is primarily designed for 1 MPI process.\n");
        }
    }
    fflush(stdout); // Ensure output is flushed

    // Initialize HYPRE
    printf("[%d] Calling HYPRE_Init...\n", myid); fflush(stdout);
    ierr = HYPRE_Init();
    HYPRE_CHECK_MSG("HYPRE_Init", ierr);
    printf("[%d] PASS: HYPRE_Init\n", myid); fflush(stdout);

    // Declare HYPRE objects
    HYPRE_StructGrid grid = NULL;
    HYPRE_StructStencil stencil = NULL;
    HYPRE_StructMatrix matrix = NULL; // Initialize to NULL

    // --- Grid Setup ---
    const int ndim = 3;
    printf("[%d] Calling HYPRE_StructGridCreate...\n", myid); fflush(stdout);
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridCreate", ierr);
    printf("[%d] PASS: HYPRE_StructGridCreate\n", myid); fflush(stdout);

    HYPRE_Int ilower[ndim] = {0, 0, 0};
    HYPRE_Int iupper[ndim] = {9, 9, 9}; // 10 points in each dim: 0..9

    // Assuming single process test - Rank 0 sets extents
    // For multi-process, each rank would set its own extents
    // if (myid == 0) { // For multi-process, remove this conditional
        printf("[%d] Calling HYPRE_StructGridSetExtents...\n", myid); fflush(stdout);
        ierr = HYPRE_StructGridSetExtents(grid, ilower, iupper);
        HYPRE_CHECK_MSG("HYPRE_StructGridSetExtents", ierr);
        printf("[%d] PASS: HYPRE_StructGridSetExtents\n", myid); fflush(stdout);
    // }

    // Barrier before assembling grid
    printf("[%d] Calling MPI_Barrier before GridAssemble...\n", myid); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Assemble the grid
    printf("[%d] Calling HYPRE_StructGridAssemble...\n", myid); fflush(stdout);
    ierr = HYPRE_StructGridAssemble(grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridAssemble", ierr);
    printf("[%d] PASS: HYPRE_StructGridAssemble\n", myid); fflush(stdout);

    // --- Stencil Setup ---
    const int stencil_size = 1;
    printf("[%d] Calling HYPRE_StructStencilCreate...\n", myid); fflush(stdout);
    ierr = HYPRE_StructStencilCreate(ndim, stencil_size, &stencil);
    HYPRE_CHECK_MSG("HYPRE_StructStencilCreate", ierr);
    printf("[%d] PASS: HYPRE_StructStencilCreate\n", myid); fflush(stdout);

    HYPRE_Int offsets[stencil_size][ndim] = {{0, 0, 0}}; // Center point only
    for (int j = 0; j < stencil_size; ++j) {
        printf("[%d] Calling HYPRE_StructStencilSetElement (element %d)...\n", myid, j); fflush(stdout);
        ierr = HYPRE_StructStencilSetElement(stencil, j, offsets[j]);
        HYPRE_CHECK_MSG("HYPRE_StructStencilSetElement", ierr);
        printf("[%d] PASS: HYPRE_StructStencilSetElement (element %d)\n", myid, j); fflush(stdout);
    }

    // --- Matrix Create and Initialize ---
    printf("[%d] Calling HYPRE_StructMatrixCreate...\n", myid); fflush(stdout);
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix);
    if (ierr == 0) {
        printf("[%d] PASS: HYPRE_StructMatrixCreate\n", myid); fflush(stdout);

        // *** Initialize the matrix ***
        printf("[%d] Calling HYPRE_StructMatrixInitialize...\n", myid); fflush(stdout);
        ierr = HYPRE_StructMatrixInitialize(matrix);
        HYPRE_CHECK_MSG("HYPRE_StructMatrixInitialize", ierr); // Will abort if init fails
        printf("[%d] PASS: HYPRE_StructMatrixInitialize\n", myid); fflush(stdout);

        // If we reach here, Create and Initialize succeeded
        // We don't Assemble or SetValues in this simple test

    } else {
        // Create failed, report it via the macro which will abort
        HYPRE_CHECK_MSG("HYPRE_StructMatrixCreate", ierr);
        final_exit_code = 1; // Should not be reached due to abort
    }

    // --- Cleanup ---
    // Destroy Matrix only if it was successfully created and initialized
    if (matrix != NULL) { // Check handle is not NULL
         printf("[%d] Calling HYPRE_StructMatrixDestroy...\n", myid); fflush(stdout);
         ierr = HYPRE_StructMatrixDestroy(matrix);
         HYPRE_CHECK_MSG("HYPRE_StructMatrixDestroy", ierr);
         printf("[%d] PASS: HYPRE_StructMatrixDestroy\n", myid); fflush(stdout);
    }

    // Destroy Stencil
    if (stencil != NULL) { // Check handle is not NULL
        printf("[%d] Calling HYPRE_StructStencilDestroy...\n", myid); fflush(stdout);
        ierr = HYPRE_StructStencilDestroy(stencil);
        HYPRE_CHECK_MSG("HYPRE_StructStencilDestroy", ierr);
        printf("[%d] PASS: HYPRE_StructStencilDestroy\n", myid); fflush(stdout);
    }

    // Destroy Grid
    if (grid != NULL) { // Check handle is not NULL
        printf("[%d] Calling HYPRE_StructGridDestroy...\n", myid); fflush(stdout);
        ierr = HYPRE_StructGridDestroy(grid);
        HYPRE_CHECK_MSG("HYPRE_StructGridDestroy", ierr);
        printf("[%d] PASS: HYPRE_StructGridDestroy\n", myid); fflush(stdout);
    }

    // Finalize HYPRE
    printf("[%d] Calling HYPRE_Finalize...\n", myid); fflush(stdout);
    ierr = HYPRE_Finalize();
    // Cannot use HYPRE_CHECK_MSG after Finalize as HYPRE error handling might be gone
    if (ierr != 0) {
         fprintf(stderr, "[%d] Warning: HYPRE_Finalize returned error code %d\n", myid, ierr);
         if (final_exit_code == 0) final_exit_code = 1; // Record error if none before
    } else {
        printf("[%d] PASS: HYPRE_Finalize\n", myid); fflush(stdout);
    }

    // Finalize MPI
    printf("[%d] Calling MPI_Finalize...\n", myid); fflush(stdout);
    MPI_Finalize();

    // Exit with appropriate code
    printf("[%d] Exiting with code %d\n", myid, final_exit_code); fflush(stdout);
    return final_exit_code;
}
