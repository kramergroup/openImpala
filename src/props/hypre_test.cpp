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
        MPI_Abort(MPI_COMM_WORLD, ierr); \
    } else { \
        printf("PASS: %s\n", msg); \
    } \
} while (0)

int main(int argc, char *argv[]) {
    int myid, num_procs;
    int ierr = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (myid == 0) {
        printf("Running HYPRE Standalone Test (hypre_test.cpp)...\n");
        printf("MPI Procs: %d\n", num_procs);
        // Note: This test is designed for -np 1, but should technically work for more.
        if (num_procs > 1) {
             printf("WARNING: This test is primarily designed for 1 MPI process.\n");
        }
    }

    // Initialize HYPRE
    // Using HYPRE_Init() is generally preferred over the old HYPRE_Init(&argc, &argv)
    ierr = HYPRE_Init();
    HYPRE_CHECK_MSG("HYPRE_Init", ierr);

    // 1. Set up Grid
    HYPRE_StructGrid grid;
    const int ndim = 3;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridCreate", ierr);

    // Define the grid extents for this process (simple 10x10x10 box on rank 0)
    // Assumes running with -np 1 for simplicity, so rank 0 owns the whole box.
    // If running with more procs, this part would need proper decomposition.
    HYPRE_Int ilower[ndim] = {0, 0, 0};
    HYPRE_Int iupper[ndim] = {9, 9, 9}; // 10 points in each dim: 0..9

    if (myid == 0) { // Only rank 0 sets extents in this simple -np 1 case
         printf("  Setting grid extents: [%d,%d,%d] to [%d,%d,%d]\n",
               ilower[0], ilower[1], ilower[2], iupper[0], iupper[1], iupper[2]);
        ierr = HYPRE_StructGridSetExtents(grid, ilower, iupper);
        HYPRE_CHECK_MSG("HYPRE_StructGridSetExtents", ierr);
    }
    // In a multi-process run, other ranks would set their extents here.
    // Barrier to ensure all ranks have potentially set extents before assembling.
    MPI_Barrier(MPI_COMM_WORLD);


    // Assemble the grid
    ierr = HYPRE_StructGridAssemble(grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridAssemble", ierr);
    printf("  Grid assembled successfully.\n");

    // 2. Set up Stencil (1-point)
    HYPRE_StructStencil stencil;
    const int stencil_size = 1;
    ierr = HYPRE_StructStencilCreate(ndim, stencil_size, &stencil);
    HYPRE_CHECK_MSG("HYPRE_StructStencilCreate", ierr);

    HYPRE_Int offsets[stencil_size][ndim] = {{0, 0, 0}}; // Center point only
    for (int j = 0; j < stencil_size; ++j) {
        ierr = HYPRE_StructStencilSetElement(stencil, j, offsets[j]);
        HYPRE_CHECK_MSG("HYPRE_StructStencilSetElement", ierr);
    }
    printf("  Stencil created successfully (1-point).\n");

    // 3. Create Matrix <-- THE CALL WE ARE TESTING
    HYPRE_StructMatrix matrix;
    printf("  Attempting HYPRE_StructMatrixCreate...\n");
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix);
    // Check the result carefully
    if (ierr != 0) {
        char hypre_error_msg[256];
        HYPRE_DescribeError(ierr, hypre_error_msg);
        fprintf(stderr, "FAIL: HYPRE_StructMatrixCreate returned error!\n");
        fprintf(stderr, "  HYPRE Error: %s (Code: %d)\n", hypre_error_msg, ierr);
        // Don't use the macro here, just report and proceed to cleanup
    } else {
        printf("PASS: HYPRE_StructMatrixCreate succeeded.\n");
        // Only destroy if create succeeded
        ierr = HYPRE_StructMatrixDestroy(matrix);
        HYPRE_CHECK_MSG("HYPRE_StructMatrixDestroy", ierr);
    }


    // Cleanup
    ierr = HYPRE_StructStencilDestroy(stencil);
    HYPRE_CHECK_MSG("HYPRE_StructStencilDestroy", ierr);
    ierr = HYPRE_StructGridDestroy(grid);
    HYPRE_CHECK_MSG("HYPRE_StructGridDestroy", ierr);

    // Finalize HYPRE
    ierr = HYPRE_Finalize();
    HYPRE_CHECK_MSG("HYPRE_Finalize", ierr);

    // Finalize MPI
    MPI_Finalize();

    // Explicitly return 0 only if the matrix create call did not fail initially
    return (ierr == 0) ? 0 : 1; // Return 0 on success, 1 on MatrixCreate failure
}
