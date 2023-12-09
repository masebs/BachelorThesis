/* Custom printing functions for the OSU benchmarks with more detailed timing for the single steps
 * Marc S. Schneider, 2023
 */

#include "osu_util_mpi.h"

#undef FIELD_WIDTH
#define FIELD_WIDTH 23

void print_detailed_times(double avg_comp_time, double avg_comm_time, 
                 double avg_decomp_time, double avg_reduce_time) {
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_comp_time);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_comm_time);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_decomp_time);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_reduce_time);
}

void print_stats_detailed (int rank, int size, int compressedSize, double avg_time, 
                           double min_time, double max_time, double avg_comp_time, 
                           double avg_comm_time, double avg_decomp_time,
                           double avg_reduce_time)
{
    if (rank) {
        return;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        print_detailed_times(avg_comp_time, avg_comm_time, avg_decomp_time, avg_reduce_time);
        fprintf(stdout, "%*d", FIELD_WIDTH, compressedSize);
    } else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        print_detailed_times(avg_comp_time, avg_comm_time, avg_decomp_time, avg_reduce_time);
        fprintf(stdout, "%*d", FIELD_WIDTH, compressedSize);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu\n",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    } else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void print_stats_validate_detailed(int rank, int size, int compressedSize, double avg_time,
            double min_time, double max_time, int errors, double avg_comp_time, 
            double avg_comm_time, double avg_decomp_time, double avg_reduce_time)
{
    if (rank) {
        return;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        print_detailed_times(avg_comp_time, avg_comm_time, avg_decomp_time, avg_reduce_time);
        fprintf(stdout, "%*d", FIELD_WIDTH, compressedSize);
    } else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        print_detailed_times(avg_comp_time, avg_comm_time, avg_decomp_time, avg_reduce_time);
        fprintf(stdout, "%*d", FIELD_WIDTH, compressedSize);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    }
    fprintf(stdout, "%*s\n", FIELD_WIDTH, VALIDATION_STATUS(errors));
    fflush(stdout);
}

void print_preamble_detailed (int rank)
{
    if (rank) {
        return;
    }

    fprintf(stdout, "\n");

    switch (options.accel) {
        case CUDA:
            printf(benchmark_header, "-CUDA");
            break;
        case OPENACC:
            printf(benchmark_header, "-OPENACC");
            break;
        case ROCM:
            printf(benchmark_header, "-ROCM");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Compression (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Communication (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Decompression (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Reduce (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg compressed length");
    } else {
        fprintf(stdout, "# Avg Latency (us)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency (us)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency (us)");
        fprintf(stdout, "%*s", 12, "Iterations");
    }

    if (options.validate)
        fprintf(stdout, "%*s", FIELD_WIDTH, "Validation");
    fprintf(stdout, "\n");
    fflush(stdout);
}
