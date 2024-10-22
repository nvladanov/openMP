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

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size=1) {
	// your work goes here, this is just a sequential code
	unsigned long visited = 0;

    // Initialize the first row and column
    #pragma omp parallel for
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;
    }

    #pragma omp parallel for
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;
    }
	
    // Compute the similarity matrix using taskloop
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (unsigned int i = 1; i < rows; i++) {
                #pragma omp taskloop grainsize(grain_size)
                for (unsigned int j = 1; j < cols; j++) {
                    float match = S[i-1][j-1] + (X[i-1] == Y[j-1] ? match_score : mismatch_score);
                    float del = S[i-1][j] + gap_penalty;
                    float insert = S[i][j-1] + gap_penalty;
                    S[i][j] = std::max({match, del, insert});

                    #pragma omp atomic
                    visited++;
                }
            }
        }
    }

    return visited;
}


unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size=1) {
	// your work goes here, this is just a sequential code
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
