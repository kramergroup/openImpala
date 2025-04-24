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
        /* Only print PASS if successful */ \
        /* printf("PASS: %s\n", msg); */ \
    } \
} while (0)

int main(int argc, char *argv[]) {
    int myid, num_procs;
    int ierr = 0;

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

    // 1. Set up Grid
    HYPRE_StructGrid grid;
    const int ndim = 3;
    printf("[%d] Calling HYPRE_StructGridCreate...\n", myid); fflush(stdout);
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridCreate", ierr);
    printf("[%d] PASS: HYPRE_StructGridCreate\n", myid); fflush(stdout);


    HYPRE_Int ilower[ndim] = {0, 0, 0};
    HYPRE_Int iupper[ndim] = {9, 9, 9}; // 10 points in each dim: 0..9

    if (myid == 0) {
        printf("[%d] Calling HYPRE_StructGridSetExtents...\n", myid); fflush(stdout);
        ierr = HYPRE_StructGridSetExtents(grid, ilower, iupper);
        HYPRE_CHECK_MSG("HYPRE_StructGridSetExtents", ierr);
        printf("[%d] PASS: HYPRE_StructGridSetExtents\n", myid); fflush(stdout);
    }
    // Barrier before assembling grid
    printf("[%d] Calling MPI_Barrier before GridAssemble...\n", myid); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);


    // Assemble the grid
    printf("[%d] Calling HYPRE_StructGridAssemble...\n", myid); fflush(stdout);
    ierr = HYPRE_StructGridAssemble(grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridAssemble", ierr);
    printf("[%d] PASS: HYPRE_StructGridAssemble\n", myid); fflush(stdout);


    // 2. Set up Stencil (1-point)
    HYPRE_StructStencil stencil;
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


    // 3. Create Matrix <-- THE CALL WE ARE TESTING
    HYPRE_StructMatrix matrix = NULL; // Initialize to NULL
    printf("[%d] Calling HYPRE_StructMatrixCreate...\n", myid); fflush(stdout);
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix);
    // Check the result carefully
    if (ierr != 0) {
        char hypre_error_msg[256];
        HYPRE_DescribeError(ierr, hypre_error_msg);
        fprintf(stderr, "[%d] FAIL: HYPRE_StructMatrixCreate returned error!\n", myid);
        fprintf(stderr, "  HYPRE Error: %s (Code: %d)\n", hypre_error_msg, ierr);
        fflush(stderr);
        // Proceed to cleanup, but record the error code to return later
    } else {
        printf("[%d] PASS: HYPRE_StructMatrixCreate\n", myid); fflush(stdout);
        // Only destroy if create succeeded and matrix is not NULL
        if (matrix != NULL) {
             printf("[%d] Calling HYPRE_StructMatrixDestroy...\n", myid); fflush(stdout);
             int destroy_ierr = HYPRE_StructMatrixDestroy(matrix);
             HYPRE_CHECK_MSG("HYPRE_StructMatrixDestroy", destroy_ierr);
             printf("[%d] PASS: HYPRE_StructMatrixDestroy\n", myid); fflush(stdout);
        } else {
             // This case should ideally not happen if HYPRE_StructMatrixCreate succeeded
             printf("[%d] Warning: Matrix handle was NULL after successful create?\n", myid); fflush(stdout);
        }
    }

    // Store the result of the critical matrix create call to return later
    int matrix_create_ierr = ierr;


    // Cleanup
    printf("[%d] Calling HYPRE_StructStencilDestroy...\n", myid); fflush(stdout);
    int stencil_destroy_ierr = HYPRE_StructStencilDestroy(stencil);
    HYPRE_CHECK_MSG("HYPRE_StructStencilDestroy", stencil_destroy_ierr);
    printf("[%d] PASS: HYPRE_StructStencilDestroy\n", myid); fflush(stdout);

    printf("[%d] Calling HYPRE_StructGridDestroy...\n", myid); fflush(stdout);
    int grid_destroy_ierr = HYPRE_StructGridDestroy(grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridDestroy", grid_destroy_ierr);
    printf("[%d] PASS: HYPRE_StructGridDestroy\n", myid); fflush(stdout);


    // Finalize HYPRE
    printf("[%d] Calling HYPRE_Finalize...\n", myid); fflush(stdout);
    int final_ierr = HYPRE_Finalize();
    HYPRE_CHECK_MSG("HYPRE_Finalize", final_ierr);
    printf("[%d] PASS: HYPRE_Finalize\n", myid); fflush(stdout);


    // Finalize MPI
    printf("[%d] Calling MPI_Finalize...\n", myid); fflush(stdout);
    MPI_Finalize();

    // Return 0 only if the matrix create call did not fail initially
    printf("[%d] Exiting with code %d\n", myid, (matrix_create_ierr == 0) ? 0 : 1); fflush(stdout);
    return (matrix_create_ierr == 0) ? 0 : 1;
}
