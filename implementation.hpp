#include <algorithm> 
#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

unsigned long SequenceInfo::gpsa_sequential(float** S) {
    unsigned long visited = 0;

	// Boundary
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
		visited++;
	}

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
		visited++;
	}
	
	// Main part
	for (unsigned int i = 1; i < rows; i++) {
		for (unsigned int j = 1; j < cols; j++) {
			float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
			float del = S[i - 1][j] + gap_penalty;
			float insert = S[i][j - 1] + gap_penalty;
			S[i][j] = std::max({match, del, insert});
		
			visited++;
		}
	}

    return visited;
}

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size) {
    unsigned long visited = 0;

    // Boundary initialization (sequential)
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }

    // Define block size
    int block_size = grain_size;

    int n_blocks_row = (rows - 1 + block_size - 1) / block_size;
    int n_blocks_col = (cols - 1 + block_size - 1) / block_size;
    int num_block_diagonals = n_blocks_row + n_blocks_col - 1;

    // Process blocks in diagonal order
    for (int k = 0; k < num_block_diagonals; k++) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                int start_i = std::max(0, k - n_blocks_col + 1);
                int end_i = std::min(k, n_blocks_row - 1);

                #pragma omp taskloop
                for (int idx = start_i; idx <= end_i; idx++) {
                    int i = idx;
                    int j = k - i;

                    int row_start = 1 + i * block_size;
                    int row_end = std::min(row_start + block_size, rows);
                    int col_start = 1 + j * block_size;
                    int col_end = std::min(col_start + block_size, cols);

                    // Compute block
                    for (int ii = row_start; ii < row_end; ii++) {
                        for (int jj = col_start; jj < col_end; jj++) {
                            float match = S[ii - 1][jj - 1] +
                                (X[ii - 1] == Y[jj - 1] ? match_score : mismatch_score);
                            float del = S[ii - 1][jj] + gap_penalty;
                            float insert = S[ii][jj - 1] + gap_penalty;
                            S[ii][jj] = std::max({match, del, insert});

                            #pragma omp atomic
                            visited++;
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size) {
    unsigned long visited = 0;

    // Boundary initialization (sequential)
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }

    // Define block size
    int block_size = grain_size;

    int n_blocks_row = (rows - 1 + block_size - 1) / block_size;
    int n_blocks_col = (cols - 1 + block_size - 1) / block_size;

    // Use a parallel region with reduction
    #pragma omp parallel reduction(+:visited)
    {
        #pragma omp single
        {
            for (int i = 0; i < n_blocks_row; i++) {
                for (int j = 0; j < n_blocks_col; j++) {
                    int row_start = 1 + i * block_size;
                    int row_end = std::min(row_start + block_size, rows);
                    int col_start = 1 + j * block_size;
                    int col_end = std::min(col_start + block_size, cols);

                    // Determine dependencies
                    int num_deps = 0;
                    void* deps_in[3];
                    if (i > 0 && j > 0) {
                        deps_in[num_deps++] = &S[row_start - 1][col_start - 1];
                    }
                    if (i > 0) {
                        deps_in[num_deps++] = &S[row_start - 1][col_start];
                    }
                    if (j > 0) {
                        deps_in[num_deps++] = &S[row_start][col_start - 1];
                    }

                    #pragma omp task depend(in: deps_in[0:num_deps])
                    {
                        unsigned long local_visited = 0;

                        // Compute block
                        for (int ii = row_start; ii < row_end; ii++) {
                            for (int jj = col_start; jj < col_end; jj++) {
                                float match = S[ii - 1][jj - 1] +
                                    (X[ii - 1] == Y[jj - 1] ? match_score : mismatch_score);
                                float del = S[ii - 1][jj] + gap_penalty;
                                float insert = S[ii][jj - 1] + gap_penalty;
                                S[ii][jj] = std::max({match, del, insert});

                                local_visited++;
                            }
                        }

                        // Reduction of local_visited to visited
                        visited += local_visited;
                    }
                }
            }
        }
    }

    return visited;
}