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

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size = 1) {
    unsigned long visited = 0;
    int gap_penalty = -2;
    int match_score = 1, mismatch_score = -1;


    unsigned int total_diagonals = rows + cols - 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (unsigned int k = 2; k <= total_diagonals - 1; k++) {
                unsigned int i_start = (std::max)(1u, k >= cols ? k - cols + 1u : 1u);
                unsigned int i_end = (std::min)(rows - 1u, k - 1u);

                #pragma omp taskloop grainsize(grain_size)
                for (unsigned int i = i_start; i <= i_end; i++) {
                    unsigned int j = k - i;
                    if (j >= 1 && j <= cols - 1) {
                        float match = S[i - 1][j - 1] + ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                        float del = S[i - 1][j] + gap_penalty;
                        float insert = S[i][j - 1] + gap_penalty;
                        S[i][j] = std::max({match, del, insert});
                        #pragma omp atomic
                        visited++;
                    }
                }
                #pragma omp taskwait // Ensure all tasks in the current anti-diagonal are completed
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