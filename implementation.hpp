#include <unordered_map>
#include <omp.h>
#include "helpers.hpp"

inline float max_of_three(float a, float b, float c) {
    float max_ab = (a > b) ? a : b;
    return (max_ab > c) ? max_ab : c;
}

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

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size=1) {
    unsigned long visited = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Parallelized Boundary Initialization using Taskloop
            #pragma omp taskloop grainsize(grain_size)
            for (unsigned int i = 1; i < rows; i++) {
                S[i][0] = i * gap_penalty;
                #pragma omp atomic
                visited++;
            }

            #pragma omp taskloop grainsize(grain_size)
            for (unsigned int j = 0; j < cols; j++) {
                S[0][j] = j * gap_penalty;
                #pragma omp atomic
                visited++;
            }

            // Main computation along anti-diagonals
            for (unsigned int k = 2; k <= rows + cols - 2; k++) {
                unsigned int start_i = (k >= cols) ? (k - cols + 1) : 1;
                unsigned int end_i = (k >= rows) ? (rows - 1) : (k - 1);

                #pragma omp taskloop grainsize(grain_size)
                for (unsigned int i = start_i; i <= end_i; i++) {
                    unsigned int j = k - i + 1;
                    float match = S[i - 1][j - 1] + ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                    float del = S[i - 1][j] + gap_penalty;
                    float insert = S[i][j - 1] + gap_penalty;
                    S[i][j] = max_of_three(match, del, insert);

                    #pragma omp atomic
                    visited++;
                }
            }
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size=1) {
    unsigned long visited = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Parallelized Boundary Initialization using Tasks
            for (unsigned int i = 1; i < rows; i += grain_size) {
                unsigned int i_end = ((i + grain_size) < rows) ? (i + grain_size) : rows;
                #pragma omp task
                {
                    for (unsigned int ii = i; ii < i_end; ++ii) {
                        S[ii][0] = ii * gap_penalty;
                        #pragma omp atomic
                        visited++;
                    }
                }
            }

            for (unsigned int j = 0; j < cols; j += grain_size) {
                unsigned int j_end = ((j + grain_size) < cols) ? (j + grain_size) : cols;
                #pragma omp task
                {
                    for (unsigned int jj = j; jj < j_end; ++jj) {
                        S[0][jj] = jj * gap_penalty;
                        #pragma omp atomic
                        visited++;
                    }
                }
            }

            // Main computation with explicit tasks
            for (unsigned int i = 1; i < rows; i++) {
                for (unsigned int j = 1; j < cols; j++) {
                    #pragma omp task depend(in: S[i-1][j-1], S[i-1][j], S[i][j-1]) depend(out: S[i][j])
                    {
                        float match = S[i - 1][j - 1] + ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                        float del = S[i - 1][j] + gap_penalty;
                        float insert = S[i][j - 1] + gap_penalty;
                        S[i][j] = max_of_three(match, del, insert);

                        #pragma omp atomic
                        visited++;
                    }
                }
            }
        }
    }

    return visited;
}