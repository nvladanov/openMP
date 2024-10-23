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
	// for (unsigned int i = 1; i < rows; i++) {
	// 	for (unsigned int j = 1; j < cols; j++) {
	// 		float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
	// 		float del = S[i - 1][j] + gap_penalty;
	// 		float insert = S[i][j - 1] + gap_penalty;
	// 		S[i][j] = std::max({match, del, insert});
		
	// 		visited++;
	// 	}
	// }

    return visited;
}

unsigned long SequenceInfo::gpsa_taskloop(float** S, int grain_size = 1) {
    unsigned long visited = 0;
    int gap_penalty = -2;  // Linear gap penalty

    // Parallelize boundary initialization with reduction for visited
    #pragma omp parallel for reduction(+:visited)
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
        visited++;  // Each thread increments its own local visited counter
    }

    #pragma omp parallel for reduction(+:visited)
    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
        visited++;  // Each thread increments its own local visited counter
    }

    // Parallelize the core loop using OpenMP taskloop and reduction
    // int match_score = 1, mismatch_score = -1;
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     {
    //         for (unsigned int k = 1; k < rows + cols - 1; k++) {
    //             #pragma omp task depend(inout: S) firstprivate(k) // Create tasks per diagonal
    //             {
    //                 unsigned int start_row = (k < rows) ? k : rows - 1;
    //                 unsigned int start_col = (k < rows) ? 1 : k - rows + 1;

    //                 for (unsigned int i = start_row, j = start_col; i >= 1 && j < cols; i--, j++) {
    //                     float match = S[i - 1][j - 1] + (X[i - 1] == Y[j - 1] ? match_score : mismatch_score);
    //                     float del = S[i - 1][j] + gap_penalty;
    //                     float insert = S[i][j - 1] + gap_penalty;
    //                     S[i][j] = std::max({match, del, insert});
                        
    //                     #pragma omp atomic
    //                     visited++;
    //                 }
    //             }
    //         }
    //     }
    // }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size = 32) {


    return 0;
}