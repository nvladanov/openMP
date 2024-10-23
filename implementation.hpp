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

    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp taskloop grainsize(512) reduction(+:visited)
        for (unsigned int i = 1; i < rows; i++) {
            S[i][0] = i * gap_penalty;
            visited++;
        }

        #pragma omp taskloop grainsize(512) reduction(+:visited)
        for (unsigned int j = 0; j < cols; j++) {
            S[0][j] = j * gap_penalty;
            visited++;
        }
    }

    unsigned int total_diagonals = rows + cols - 1;

    #pragma omp parallel reduction(+:visited)
    {
        #pragma omp single
        {
            for (unsigned int k = 2; k <= total_diagonals - 1; k++) {
                unsigned int i_start = (k <= cols - 1) ? 1 : k - (cols - 1);
                unsigned int i_end   = (k <= rows - 1) ? k - 1 : rows - 1;

                #pragma omp taskloop grainsize(grain_size) reduction(+:visited)
                for (unsigned int i = i_start; i <= i_end; i++) {
                    unsigned int j = k - i;
                    if (j >= 1 && j <= cols - 1) {
                        float match = S[i - 1][j - 1] + ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                        float del = S[i - 1][j] + gap_penalty;
                        float insert = S[i][j - 1] + gap_penalty;
                        S[i][j] = std::max({match, del, insert});
                        visited++;
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    return visited;
}

unsigned long SequenceInfo::gpsa_tasks(float** S, int grain_size = 1) {
    unsigned long total_visited = 0;

    // Boundary initialization (sequential)
    for (unsigned int i = 1; i < rows; i++) {
        S[i][0] = i * gap_penalty;
    }

    for (unsigned int j = 0; j < cols; j++) {
        S[0][j] = j * gap_penalty;
    }

    unsigned int total_diagonals = rows + cols - 1;
    int* deps = new int[total_diagonals];
    for (unsigned int i = 0; i < total_diagonals; i++) deps[i] = 0;

    // Use thread-private variables for visited counts
    unsigned long* visited_counts = new unsigned long[omp_get_max_threads()]();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp single
        {
            for (unsigned int k = 2; k <= total_diagonals - 1; k++) {
                unsigned int i_start = (std::max)(1u, k >= cols ? k - cols + 1u : 1u);
                unsigned int i_end = (std::min)(rows - 1u, k - 1u);

                for (unsigned int i_block = i_start; i_block <= i_end; i_block += grain_size) {
                    unsigned int i_block_end = std::min(i_block + grain_size - 1, i_end);

                    #pragma omp task firstprivate(k, i_block, i_block_end, thread_id) \
                        depend(in: deps[k - 1]) depend(inout: deps[k])
                    {
                        unsigned long local_visited = 0;
                        for (unsigned int i = i_block; i <= i_block_end; i++) {
                            unsigned int j = k - i;
                            if (j >= 1 && j <= cols - 1) {
                                float match = S[i - 1][j - 1] + ((X[i - 1] == Y[j - 1]) ? match_score : mismatch_score);
                                float del = S[i - 1][j] + gap_penalty;
                                float insert = S[i][j - 1] + gap_penalty;
                                S[i][j] = std::max({match, del, insert});
                                local_visited++;
                            }
                        }
                        // Accumulate local visited count to thread's total
                        visited_counts[thread_id] += local_visited;
                    }
                }
            }
        }
    }

    // Sum up the visited counts from all threads
    for (int i = 0; i < omp_get_max_threads(); i++) {
        total_visited += visited_counts[i];
    }

    delete[] deps;
    delete[] visited_counts;
    return total_visited;
}