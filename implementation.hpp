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

unsigned long SequenceInfo::gpsa_taskloop(float** S, int block_size = 1) {
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

     // Calculate the number of blocks in each dimension
    int num_blocks_x = (rows + block_size - 1) / block_size;
    int num_blocks_y = (cols + block_size - 1) / block_size;
	
	// Main part
   #pragma omp parallel
    {
        #pragma omp single
        {
            // Iterate over the antidiagonals of blocks
            for (unsigned int k = 0; k < num_blocks_x + num_blocks_y - 1; k++) {
                #pragma omp taskloop reduction(+:visited)
                for (unsigned int block_i = std::max(0, (int)(k - num_blocks_y + 1)); block_i <= std::min(k, (unsigned int)num_blocks_x - 1); block_i++) {
                    unsigned int block_j = k - block_i;

                    // For each block (block_i, block_j), process the wavefront inside the block
                    unsigned int start_row = block_i * block_size + 1;
                    unsigned int end_row = std::min(static_cast<int>((block_i + 1) * block_size), rows);
                    unsigned int start_col = block_j * block_size + 1;
                    unsigned int end_col = std::min(static_cast<int>((block_j + 1) * block_size), cols);

                    for (unsigned int diag = 0; diag < (end_row - start_row + 1) + (end_col - start_col + 1) - 1; diag++) {
                        for (unsigned int i = std::min(start_row + diag, end_row); i > start_row && (i - start_row) < (end_col - start_col); i--) {
                            unsigned int j = start_col + diag - (i - start_row);
                            if (j < end_col) {
                                float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                                float del = S[i - 1][j] + gap_penalty;
                                float insert = S[i][j - 1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});
                                visited++;  // Increment visited count
                            }
                        }
                    }
                }
            }
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size = 32) {
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