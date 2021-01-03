#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "chag/pp/prefix.cuh"
#include "chag/pp/reduce.cuh"

//Image loading
#include <uc/img/img.h>
#include <uc/util/utf8_conv.h>
#include <uc/os/windows/com_initializer.h>

namespace pp = chag::pp;

namespace image
{
    struct yuv_image
    {
        std::vector<uint8_t> m_pixels;
        uint32_t             m_width;
        uint32_t             m_height;
        uint32_t             m_row_pitch;

        uint8_t* y_plane()
        {
            return &m_pixels[0];
        }

        uint8_t* u_plane()
        {
            const uint32_t  w   = m_width;
            const uint32_t  h   = m_height;

            uint8_t* y_plane    = &m_pixels[0];
            uint8_t* u_plane    = &m_pixels[0] + size_t(w) * size_t(h);
            uint8_t* v_plane    = &m_pixels[0] + size_t(2) * size_t(w) * size_t(h);

            return u_plane;
        }

        uint8_t* v_plane()
        {
            const uint32_t  w = m_width;
            const uint32_t  h = m_height;

            uint8_t* y_plane = &m_pixels[0];
            uint8_t* u_plane = &m_pixels[0] + size_t(w) * size_t(h);
            uint8_t* v_plane = &m_pixels[0] + size_t(2) * size_t(w) * size_t(h);

            return v_plane;
        }

        size_t plane_size() const
        {
            const uint32_t  w = m_width;
            const uint32_t  h = m_height;
            return size_t(w) * size_t(h);
        }
    };

    inline uint8_t clip(uint8_t x)
    {
        return ((x) > 255 ? 255 : (x) < 0 ? 0 : x);
    }

    inline uint8_t rgb2y(uint8_t r, uint8_t g, uint8_t b)
    {
        return clip(((66 * (r)+129 * (g)+25 * (b)+128) >> 8) + 16);
    }

    inline uint8_t rgb2u(uint8_t r, uint8_t g, uint8_t b)
    {
        return clip(((-38 * (r)-74 * (g)+112 * (b)+128) >> 8) + 128);
    }

    inline uint8_t rgb2v(uint8_t r, uint8_t g, uint8_t b)
    {
        return clip(((112 * (r)-94 * (g)-18 * (b)+128) >> 8) + 128);
    }

    yuv_image to_yuv(const uc::gx::imaging::cpu_texture& bgra)
    {
        yuv_image r;
        const uint32_t  w = bgra.width();
        const uint32_t  h = bgra.height();
        r.m_pixels.resize(size_t(w) * size_t(h) * 3U);

        r.m_width = w;
        r.m_height = h;
        r.m_row_pitch = bgra.row_pitch();

        uint8_t* y_plane = &r.m_pixels[0];
        uint8_t* u_plane = &r.m_pixels[0] + size_t(w) * size_t(h);
        uint8_t* v_plane = &r.m_pixels[0] + size_t(2) * size_t(w) * size_t(h);

        const auto p = bgra.pixels();

        for (auto i = 0U; i < h; ++i)
        {
            const uint8_t* row = p.get_pixels_cpu() + size_t(i) * bgra.row_pitch();
            uint8_t* y_row = y_plane + size_t(i) * size_t(w);
            uint8_t* u_row = u_plane + size_t(i) * size_t(w);
            uint8_t* v_row = v_plane + size_t(i) * size_t(w);

            for (auto j = 0U; j < w; ++j)
            {
                uint8_t b = row[4 * j];
                uint8_t g = row[4 * j + 1];
                uint8_t r = row[4 * j + 2];
                uint8_t a = row[4 * j + 3];

                uint8_t* y = y_row + size_t(j);
                uint8_t* u = u_row + size_t(j);
                uint8_t* v = v_row + size_t(j);

                *y = rgb2y(r, g, b);
                *u = rgb2u(r, g, b);
                *v = rgb2v(r, g, b);
            }
        }

        return r;
    }
}

struct rle_comprression
{
    // global host memory arrays.
    uint8_t*     g_symbolsOut;
    int*         g_countsOut;
    uint8_t*     g_in;

    // Device memory used in PARLE
    uint8_t*    d_symbolsOut;
    int*        d_countsOut;
    uint8_t*    d_in;

    int32_t*    d_totalRuns;
    int32_t*    d_backwardMask;
    int32_t*    d_scannedBackwardMask;
    int32_t*    d_compactedBackwardMask;
};

rle_comprression make_compression(size_t size)
{
    rle_comprression r;

    // allocate resources on device. These arrays are used globally thoughouts the program.
    CUDA_CHECK(cudaMalloc((void**)&r.d_backwardMask, size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&r.d_scannedBackwardMask, size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&r.d_compactedBackwardMask, (size + 1) * sizeof(int32_t)));

    CUDA_CHECK(cudaMalloc((void**)&r.d_in, size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&r.d_countsOut, size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&r.d_symbolsOut, size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&r.d_totalRuns, sizeof(int32_t)));

    return r;
}

void free_compression(rle_comprression r)
{
    CUDA_CHECK(cudaFree(r.d_backwardMask));
    CUDA_CHECK(cudaFree(r.d_scannedBackwardMask));
    CUDA_CHECK(cudaFree(r.d_compactedBackwardMask));
    CUDA_CHECK(cudaFree(r.d_in));
    CUDA_CHECK(cudaFree(r.d_countsOut));
    CUDA_CHECK(cudaFree(r.d_symbolsOut));
    CUDA_CHECK(cudaFree(r.d_totalRuns));
}

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

void parleDevice(int *d_in, int n,
    int* d_symbolsOut,
    int* d_countsOut,
    int* d_totalRuns
    );

int parleHost(int *h_in, int n,
    int* h_symbolsOut,
    int* h_countsOut);

int rleCpu(uint8_t *in, int n,
    uint8_t* symbolsOut,
    int* countsOut);

__global__ void compactKernel(int* g_scannedBackwardMask, int* g_compactedBackwardMask, int* g_totalRuns, int n) {
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

__global__ void scatterKernel(int* g_compactedBackwardMask, int* g_totalRuns, uint8_t* g_in, uint8_t* g_symbolsOut, int* g_countsOut) {
    int n = *g_totalRuns;

    for (int i : hemi::grid_stride_range(0, n)) {
        int a = g_compactedBackwardMask[i];
        int b = g_compactedBackwardMask[i + 1];

        g_symbolsOut[i] = g_in[a];
        g_countsOut[i] = b - a;
    }
}

__global__ void maskKernel(uint8_t *g_in, int* g_backwardMask, int n)
{
    for (int i : hemi::grid_stride_range(0, n))
    {
        if (i == 0)
            g_backwardMask[i] = 1;
        else {
            g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
        }
    }
}

void PrintArray(int* arr, int n)
{
    for (int i = 0; i < n; ++i){
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

char errorString[256];

bool verifyCompression(
    uint8_t* original, int n,
    uint8_t* compressedSymbols, int* compressedCounts, int totalRuns, uint8_t* g_temp_buffer)
{
    // decompress.
    int j = 0;

    int sum = 0;
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
            g_temp_buffer[j++] = symbol;
        }
    }

    // verify the compression.
    for (int i = 0; i < n; ++i)
    {
        if (original[i] != g_temp_buffer[i])
        {
            sprintf(errorString, "Decompressed and original not equal at %d, %d != %d\n", i, original[i], g_temp_buffer[i]);
            return false;
        }
    }

    return true;
}

/*
// profile some RLE implementation on the GPU.
template<typename F, typename G>
void profileGpu(F rle, G dataGen) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < NUM_TESTS; ++i) {

        int n = Tests[i];
        int* in = dataGen(n);

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

            int* in = generateCompressibleRandomData(n);

            unitTest(in, n, f, true);
        }
        printf("-------------------------------\n\n");
    }
}
*/

int main(){

    image::yuv_image yuv;
    {
        try
        {
            using namespace uc::gx::imaging;
            uc::os::windows::com_initializer c;

            const std::wstring w = L"data\\test-image.png";
            const auto image = read_image(w.c_str());

            yuv = image::to_yuv(image);
        }

        catch (...)
        {
            return -1;
        }
    }

    //compress y
    {
        std::vector<uint8_t> symbols_out(yuv.plane_size() * 2);
        std::vector<int32_t> symbols_count(yuv.plane_size() * 2);
        std::vector<uint8_t> work_buffer(yuv.plane_size() * 2);

        int32_t symbols_y = rleCpu(yuv.y_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0]);
        verifyCompression(yuv.y_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0], symbols_y, &work_buffer[0]);
    }

    //compress u
    {
        std::vector<uint8_t> symbols_out(yuv.plane_size() * 2);
        std::vector<int32_t> symbols_count(yuv.plane_size() * 2);
        std::vector<uint8_t> work_buffer(yuv.plane_size() * 2);
        int32_t symbols_u = rleCpu(yuv.u_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0]);
        verifyCompression(yuv.u_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0], symbols_u, &work_buffer[0]);
    }

    //compress v
    {
        std::vector<uint8_t> symbols_out(yuv.plane_size() * 2);
        std::vector<int32_t> symbols_count(yuv.plane_size() * 2);
        std::vector<uint8_t> work_buffer(yuv.plane_size() * 2);

        int32_t symbols_v = rleCpu(yuv.v_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0]);
        verifyCompression(yuv.v_plane(), yuv.plane_size(), &symbols_out[0], &symbols_count[0], symbols_v, &work_buffer[0]);
    }

    srand(1000);
    CUDA_CHECK(cudaSetDevice(0));

    rle_comprression compression_y = make_compression(yuv.plane_size() * 2);
    rle_comprression compression_u = make_compression(yuv.plane_size() * 2);
    rle_comprression compression_v = make_compression(yuv.plane_size() * 2);

    // transfer input data to device.
    CUDA_CHECK(cudaMemcpy(compression_y.d_in, yuv.y_plane(), yuv.plane_size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compression_u.d_in, yuv.u_plane(), yuv.plane_size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compression_v.d_in, yuv.v_plane(), yuv.plane_size(), cudaMemcpyHostToDevice));

    int32_t h_totalRuns_y;
    int32_t h_totalRuns_u;
    int32_t h_totalRuns_v;




    std::vector<uint8_t> symbols_out_y(yuv.plane_size() * 2);
    std::vector<uint8_t> symbols_out_u(yuv.plane_size() * 2);
    std::vector<uint8_t> symbols_out_v(yuv.plane_size() * 2);

    std::vector<int32_t> symbols_count_y(yuv.plane_size() * 2);
    std::vector<int32_t> symbols_count_u(yuv.plane_size() * 2);
    std::vector<int32_t> symbols_count_v(yuv.plane_size() * 2);

    CUDA_CHECK(cudaMemcpy(&h_totalRuns_y, compression_y.d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_totalRuns_y, compression_u.d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_totalRuns_v, compression_v.d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));




    // transer result data to host.
    CUDA_CHECK(cudaMemcpy(&symbols_out_y[0],   compression_y.d_symbolsOut,   h_totalRuns_y * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&symbols_count_y[0], compression_y.d_countsOut,    h_totalRuns_y * sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&symbols_out_u[0], compression_u.d_symbolsOut, h_totalRuns_u * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&symbols_count_u[0], compression_u.d_countsOut, h_totalRuns_u * sizeof(int32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&symbols_out_v[0], compression_v.d_symbolsOut, h_totalRuns_v * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&symbols_count_v[0], compression_v.d_countsOut, h_totalRuns_v * sizeof(int32_t), cudaMemcpyDeviceToHost));
   


    free_compression(compression_y);
    free_compression(compression_u);
    free_compression(compression_v);

    return 0;
}

// implementation of RLE on CPU.
int rleCpu(uint8_t *in, int n, uint8_t* symbolsOut, int* countsOut){

    if (n == 0)
        return 0; // nothing to compress!

    int outIndex = 0;
    int symbol = in[0];
    int count = 1;

    for (int i = 1; i < n; ++i)
    {
        if (in[i] != symbol)
        {
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
/*
// On the CPU do preparation to run parle, launch PARLE on GPU, and then transfer the result data to the CPU. 
int parleHost(int *h_in, int n,
    int* h_symbolsOut,
    int* h_countsOut){

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
*/

void scan(int* d_in, int* d_out, int N) {
    pp::prefix_inclusive(d_in, d_in + N, d_out);
}

// run parle on the GPU
void parleDevice(
    uint8_t *d_in, 
    int n,
    
    uint8_t* d_symbolsOut,
    int* d_countsOut,
    int* d_totalRuns,

    int* d_backwardMask,
    int* d_scannedBackwardMask,
    int* d_compactedBackwardMask
    )
{
    hemi::cudaLaunch(maskKernel, d_in, d_backwardMask, n);
    scan(d_backwardMask, d_scannedBackwardMask, n);
    hemi::cudaLaunch(compactKernel, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);
    hemi::cudaLaunch(scatterKernel, d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
}


