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

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size = 64) {
    unsigned long visited = 0;
    int gap_penalty = -2;  // Linear gap penalty

    // Initialize boundary conditions (sequential)
    #pragma omp parallel for
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        // #pragma omp atomic
        // visited++;
    }

    #pragma omp parallel for
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        // #pragma omp atomic
        // visited++;
    }

    // Parallelize the core loop using OpenMP taskloop
    int match_score = 1, mismatch_score = -1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp taskloop grainsize(grain_size = 32)
            for (unsigned int i = 1; i < rows; i++) {
                for (unsigned int j = 1; j < cols; j++) {
                    float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                    float del = S[i - 1][j] + gap_penalty;
                    float insert = S[i][j - 1] + gap_penalty;
                    S[i][j] = std::max({match, del, insert});
                    // #pragma omp atomic
                    // visited++;
                }
            }
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size = 32) {
    unsigned long visited = 0;
    int gap_penalty = -2;  // Linear gap penalty

    // Initialize boundary conditions (sequential)
    #pragma omp parallel for
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        #pragma omp atomic
        visited++;
    }

    #pragma omp parallel for
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        #pragma omp atomic
        visited++;
    }

    // Use explicit tasks for parallelization with dependencies
    int match_score = 1, mismatch_score = -1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (unsigned int i = 1; i < rows; i++) {
                for (unsigned int j = 1; j < cols; j++) {
                    #pragma omp task depend(in: S[i-1][j], S[i][j-1], S[i-1][j-1]) depend(out: S[i][j])
                    {
                        float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
                        float del = S[i - 1][j] + gap_penalty;
                        float insert = S[i][j - 1] + gap_penalty;
                        S[i][j] = std::max({match, del, insert});
                        #pragma omp atomic
                        visited++;
                    }
                }
            }
        }
    }

    return visited;
}