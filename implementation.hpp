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
    int gap_penalty = -2;
    int match_score = 1, mismatch_score = -1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(rows/64) reduction(+:visited)
            for (int i = 1; i < rows; i++) {
                S[0][i] = i * gap_penalty;
                visited++;
            }

            #pragma omp taskloop grainsize(cols/64) reduction(+:visited)
            for (int j = 0; j < cols; j++) {
                S[j][0] = j * gap_penalty;
                visited++;
            }
        }
    }


    int num_blocks_x = (rows + block_size - 1) / block_size;
    int num_blocks_y = (cols + block_size - 1) / block_size;
    
    for (int block_row = 0; block_row < num_blocks_x; block_row++) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp taskloop reduction(+:visited)
                for (int block_col = 0; block_col <  num_blocks_y; block_col++) {
                        // Параллельная обработка блоков в рамках одной горизонтали
                        // Определяем границы блока
                        int start_row = block_row * block_size + 1;
                        int end_row = std::min((block_row + 1) * block_size, rows);
                        int start_col = block_col * block_size + 1;
                        int end_col = std::min((block_col + 1) * block_size, cols);

                        // Вычисления внутри каждого блока по диагональной схеме
                        #pragma omp taskloop reduction(+:visited)
                        for (int diag = 0; diag < (end_row - start_row + 1) + (end_col - start_col + 1) - 1; diag++) {
                            for (int i = std::min(start_row + diag, end_row); i > start_row && (i - start_row) < (end_col - start_col); i--) {
                                int j = start_col + diag - (i - start_row);
                                if (j < end_col) {
                                    float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                                    float del = S[i - 1][j] + gap_penalty;
                                    float insert = S[i][j - 1] + gap_penalty;
                                    S[i][j] = std::max({match, del, insert});
                                    visited++;  // Счётчик посещённых ячеек
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