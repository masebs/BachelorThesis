/* Custom printing functions for the OSU benchmarks with more detailed timing for the single steps
 * Marc S. Schneider, 2023
 */
 
void print_stats_detailed (int rank, int size, int compressedSize, double avg_time, 
                           double min_time, double max_time, double avg_comp_time, 
                           double avg_comm_time, double avg_decomp_time,
                           double avg_reduce_time);

void print_stats_validate_detailed(int rank, int size, int compressedSize,
            double avg_time, double min_time, double max_time, int errors, 
            double avg_comp_time, double avg_comm_time, 
            double avg_decomp_time, double avg_reduce_time);

void print_preamble_detailed(int rank);
