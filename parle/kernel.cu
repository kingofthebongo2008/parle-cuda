#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "chag/pp/prefix.cuh"
#include "chag/pp/reduce.cuh"

#include <uc/img/img.h>
#include <uc/os/windows/com_initializer.h>
#include <string>
#include <iostream>

namespace pp = chag::pp;

// global host memory arrays.
uint32_t* g_symbolsOut;
uint32_t* g_countsOut;
uint32_t* g_in;
uint32_t* g_decompressed;

// Device memory used in PARLE
uint32_t* d_symbolsOut;
uint32_t* d_countsOut;
uint32_t* d_in;
uint32_t* d_totalRuns;
uint32_t* d_backwardMask;
uint32_t* d_scannedBackwardMask;
uint32_t* d_compactedBackwardMask;

const int NUM_TESTS = 11;
const int Tests[NUM_TESTS] = {
    10000, // 10K
    50000, // 50K
    100000, // 100K
    200000, // 200K
    500000, // 500K
    1000000, // 1M
    2000000, // 2M
    5000000, // 5M
    10000000, // 10M
    20000000, // 20M
    40000000, // 40M
};

const int PROFILING_TESTS = 100;
const int MAX_N = 1 << 26; // max size of any array that we use.

void parleDevice(uint32_t*d_in, int32_t n,
    uint32_t* d_symbolsOut,
    uint32_t* d_countsOut,
    uint32_t* d_totalRuns
    );

int parleHost(uint32_t*h_in, int32_t n, uint32_t* h_symbolsOut, uint32_t* h_countsOut);
int rleCpu(uint32_t *in, int32_t n, uint32_t* symbolsOut, uint32_t* countsOut);

__global__ void compactKernel(uint32_t* g_in, uint32_t* g_scannedBackwardMask, uint32_t* g_compactedBackwardMask, uint32_t* g_totalRuns, int32_t n)
{
    for (int i : hemi::grid_stride_range(0, n)) {

        if (i == (n - 1)) {
            g_compactedBackwardMask[g_scannedBackwardMask[i] + 0] = i + 1;
            *g_totalRuns = g_scannedBackwardMask[i];
        }

        if (i == 0) {
            g_compactedBackwardMask[0] = 0;
        }
        else if (g_scannedBackwardMask[i] != g_scannedBackwardMask[i - 1]) {
            g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i;
        }
    }
}

__global__ void scatterKernel(uint32_t* g_compactedBackwardMask, uint32_t* g_totalRuns, uint32_t* g_in, uint32_t* g_symbolsOut, uint32_t* g_countsOut) {
    int n = *g_totalRuns;

    for (int i : hemi::grid_stride_range(0, n))
    {
        int a = g_compactedBackwardMask[i];
        int b = g_compactedBackwardMask[i + 1];

        g_symbolsOut[i] = g_in[a];
        g_countsOut[i] = b - a;
    }
}

__global__ void maskKernel(uint32_t* g_in, uint32_t* g_backwardMask, int n) {
    for (int i : hemi::grid_stride_range(0, n)) {
        if (i == 0)
            g_backwardMask[i] = 1;
        else {
            g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
        }
    }
}

void PrintArray(uint32_t* arr, int n){
    for (int i = 0; i < n; ++i){
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

char errorString[256];

bool verifyCompression(
    uint32_t* original, int n,
    uint32_t* compressedSymbols, uint32_t* compressedCounts, int totalRuns){

    // decompress.
    int j = 0;

    uint32_t sum = 0;
    for (int i = 0; i < totalRuns; ++i) {
        sum += compressedCounts[i];
    }

    if (sum != n) {
        sprintf(errorString, "Decompressed and original size not equal %d != %d\n", n, sum);

        for (int i = 0; i < totalRuns; ++i){
            int symbol = compressedSymbols[i];
            int count = compressedCounts[i];

            printf("%d, %d\n", count, symbol);
        }
        return false;
    }

    for (int i = 0; i < totalRuns; ++i){
        int symbol = compressedSymbols[i];
        int count = compressedCounts[i];

        for (int k = 0; k < count; ++k){
            g_decompressed[j++] = symbol;
        }
    }

    // verify the compression.
    for (int i = 0; i < n; ++i) {
        if (original[i] != g_decompressed[i]){

            sprintf(errorString, "Decompressed and original not equal at %d, %d != %d\n", i, original[i], g_decompressed[i]);
            return false;
        }
    }

    return true;
}

// get random test data for compression.
// the kind of data generated is like
// 1,1,1,1,4,4,4,4,7,7,7,7,....
// so there's lots of repeated sequences. 
uint32_t* generateCompressibleRandomData(int n){
    int val = rand() % 10;

    for (int i = 0; i < n; ++i) {
        g_in[i] = val;

        if (rand() % 6 == 0){
            val = rand() % 10;
        }
    }
    return g_in;
}


// get random test data for compression.
// the kind of data generated is like
// 1,5,8,4,2,6,....
// so it's completely random.
uint32_t* generateRandomData(int n){
    for (int i = 0; i < n; ++i) {
        g_in[i] = rand() % 10;;

    }
    return g_in;
}


// use f to RLE compresss the data, and then verify the compression. 
template<typename F>
void unitTest(uint32_t* in, int n, F f, bool verbose)
{
    int totalRuns = f(in, n, g_symbolsOut, g_countsOut);

    if (verbose) {
        printf("n = %d\n", n);
        printf("Original Size  : %d\n", n);
        printf("Compressed Size: %d\n", totalRuns * 2);
    }

    if (!verifyCompression(
        in, n,
        g_symbolsOut, g_countsOut, totalRuns)) {
        printf("Failed test %s\n", errorString);
        PrintArray(in, n);

        exit(1);
    }
    else {
        if (verbose)
            printf("passed test!\n\n");
    }
}

// profile some RLE implementation on the CPU.
template<typename F, typename G>
void profileCpu(F rle, G dataGen) {
    for (int i = 0; i < NUM_TESTS; ++i) {
        int n = Tests[i];
        uint32_t* in = dataGen(n);

        StartCounter();

        for (int i = 0; i < PROFILING_TESTS; ++i) {
            rle(in, n, g_symbolsOut, g_countsOut);
        }
        printf("For n = %d, in time %.5f microseconds\n", n, (GetCounter() / ((float)PROFILING_TESTS)) * 1000.0f);

        // also unit test, to make sure that the compression is valid.
        unitTest(in, n, rle, false);
    }
}

// profile some RLE implementation on the GPU.
template<typename F, typename G>
void profileGpu(F rle, G dataGen) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < NUM_TESTS; ++i) {

        int n = Tests[i];
        uint32_t* in = dataGen(n);

        // transer input data to device.
        CUDA_CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));

        // record.
        cudaEventRecord(start);
        for (int i = 0; i < PROFILING_TESTS; ++i) {
            parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        // also unit test, to make sure that the compression is valid.
        unitTest(in, n, rle, false);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("For n = %d, in time %.5f microseconds\n", n, (ms / ((float)PROFILING_TESTS)) *1000.0f);
    }
}

// Run many unit tests on an implementation(f) of RLE.
template<typename F>
void runTests(int a, F f) {
    printf("START UNIT TESTS\n");

    for (int i = 4; i < a; ++i) {
        for (int k = 0; k < 30; ++k) {
            int n = 2 << i;

            if (k != 0) {
                // in first test, do with nice values for 'n'
                // on the other two tests, do with slightly randomized values.
                n = (int)(n * (0.6f + 1.3f * (rand() / (float)RAND_MAX)));
            }

            uint32_t* in = generateCompressibleRandomData(n);

            unitTest(in, n, f, true);
        }
        printf("-------------------------------\n\n");
    }
}

int main(){

    /*
    {
        try
        {
            using namespace uc::gx::imaging;
            uc::os::windows::com_initializer c;
            
            const std::wstring w = L"data\\test-image.png";
            const auto image     = read_image(w.c_str());
            const auto bytes     = image.pixels().get_pixels_cpu();
            const auto pitch     = image.row_pitch();
            const auto width     = image.width() * 4;

            #if defined(EXPORT_DATA)
            for (auto i = 0; i < image.size(); ++i)
            {
                if (i % pitch == 0)
                {
                    std::cout << "\n";
                }

                int32_t b = bytes[i];
                std::cout << b << ", ";
            }
            #endif
        }

        catch (...)
        {
            return -1;
        }
    }

    return 0;
    */

    srand(1000);
    CUDA_CHECK(cudaSetDevice(0));

    // allocate resources on device. These arrays are used globally thoughouts the program.
    CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, MAX_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, MAX_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_compactedBackwardMask, (MAX_N + 1) * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**)&d_in, MAX_N* sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_countsOut, MAX_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, MAX_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(uint32_t)));

    // allocate resources on the host. 
    g_in = new uint32_t[MAX_N];
    g_decompressed = new uint32_t[MAX_N];
    g_symbolsOut = new uint32_t[MAX_N];
    g_countsOut = new uint32_t[MAX_N];

    // We run this code to run many unit tests on the code
    /*
    runTests(21, rleCpu);
    runTests(21, parleHost);
    */

    // We run this code to profile the performance. 

    printf("profile random CPU\n");
    profileCpu(rleCpu, generateRandomData);

    printf("profile compressible CPU\n");
    profileCpu(rleCpu, generateCompressibleRandomData);

    printf("profile random GPU\n");
    profileGpu(parleHost, generateRandomData);

    printf("profile compressible GPU\n");
    profileGpu(parleHost, generateCompressibleRandomData);

    // We run this code when we wish to run NVPP on the algorithm. 
    /*
    int n = 1 << 23;
    unitTest(generateCompressibleRandomData(1<<23), n, rleGpu, true);
    */

    // free device arrays.
    CUDA_CHECK(cudaFree(d_backwardMask));
    CUDA_CHECK(cudaFree(d_scannedBackwardMask));
    CUDA_CHECK(cudaFree(d_compactedBackwardMask));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_countsOut));
    CUDA_CHECK(cudaFree(d_symbolsOut));
    CUDA_CHECK(cudaFree(d_totalRuns));

    CUDA_CHECK(cudaDeviceReset());

    // free host memory.
    delete[] g_in;
    delete[] g_decompressed;

    delete[] g_symbolsOut;
    delete[] g_countsOut;

    return 0;
}



// implementation of RLE on CPU.
int rleCpu(uint32_t*in, int32_t n, uint32_t* symbolsOut, uint32_t* countsOut){

    if (n == 0)
        return 0; // nothing to compress!

    uint32_t outIndex = 0;
    uint32_t symbol = in[0];
    uint32_t count = 1;

    for (uint32_t i = 1U; i < n; ++i) {
        if (in[i] != symbol) {
            // run is over.
            // So output run.
            symbolsOut[outIndex] = symbol;
            countsOut[outIndex] = count;
            outIndex++;

            // and start new run:
            symbol = in[i];
            count = 1;
        }
        else
        {
            ++count; // run is not over yet.
        }
    }

    // output last run. 
    symbolsOut[outIndex] = symbol;
    countsOut[outIndex] = count;
    outIndex++;

    return outIndex;
}

// On the CPU do preparation to run parle, launch PARLE on GPU, and then transfer the result data to the CPU. 
int parleHost(uint32_t *h_in, int n,
    uint32_t* h_symbolsOut,
    uint32_t* h_countsOut){

    int h_totalRuns;

    // transer input data to device.
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));

    // RUN.    
    parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);

    // transer result data to host.
    CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));

    return h_totalRuns;
}

void scan(uint32_t* d_in, uint32_t* d_out, int N) {
    pp::prefix_inclusive(d_in, d_in + N, d_out);
}

// run parle on the GPU
void parleDevice(uint32_t*d_in, int32_t n,
    uint32_t* d_symbolsOut,
    uint32_t* d_countsOut,
    uint32_t* d_totalRuns
    ){
    hemi::cudaLaunch(maskKernel, d_in, d_backwardMask, n);
    scan(d_backwardMask, d_scannedBackwardMask, n);
    hemi::cudaLaunch(compactKernel, d_in, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);
    hemi::cudaLaunch(scatterKernel, d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
}


