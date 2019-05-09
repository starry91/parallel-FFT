#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono> 


using namespace std::chrono;

//#include <cv.h>
//#include <highgui.h>

using namespace std;
using namespace cv;

typedef complex<float> base;
typedef float2 Complex_my;

template <typename T>
ostream &operator<<(ostream &o, vector<T> v)
{
    if (v.size() > 0)
        o << v[0];
    for (unsigned i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}
static __device__ __host__ inline Complex_my Add(Complex_my A, Complex_my B)
{
    Complex_my C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

/**
 *  Inverse of Complex_my Number
 */
static __device__ __host__ inline Complex_my Inverse(Complex_my A)
{
    Complex_my C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

/**
 *  Multipication of Complex_my Numbers
 */
static __device__ __host__ inline Complex_my Multiply(Complex_my A, Complex_my B)
{
    Complex_my C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
}

/**
* Parallel Functions for performing various tasks
*/

/**
*  Dividing by constant for inverse fft transform
*/
__global__ void inplace_divide_invert(Complex_my *A, int n, int threads)
{
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n)
    {
        // printf("in divide");
        A[i].x /= n;
        A[i].y /= n;
    }
    else
    {
        // printf("else in divide");
        // printf("i=%d, n=%d", i, n);
    }
}

/**
* Reorders array by bit-reversing the indexes.
*/
__global__ void bitrev_reorder(Complex_my *__restrict__ r, Complex_my *__restrict__ d, int s, size_t nthr, int n)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    //r[id].x = -1;
    if (id < n and __brev(id) >> (32 - s) < n)
        r[__brev(id) >> (32 - s)] = d[id];
}

/**
* Inner part of the for loop
*/
__device__ void inplace_fft_inner(Complex_my *__restrict__ A, int i, int j, int len, int n, bool invert)
{
    if (i + j + len / 2 < n and j < len / 2)
    {
        Complex_my u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? -1.0 : 1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        // printf("i:%d j:%d u_x:%f u_y:%f    v_x:%f v_y:%f\n", i, j, u.x, u.y, v.x, v.y);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

/**
* FFT if number of threads are sufficient.
*/
__global__ void inplace_fft(Complex_my *__restrict__ A, int i, int len, int n, int threads, bool invert)
{
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}

/**
* FFt if number of threads are not sufficient.
*/
__global__ void inplace_fft_outer(Complex_my *__restrict__ A, int len, int n, int threads, bool invert)
{
    int i = (blockIdx.x * threads + threadIdx.x);
    for (int j = 0; j < len / 2; j++)
    {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}

/**
* parallel FFT transform and inverse transform
* Arguments vector of complex numbers, invert, balance, number of threads
* Perform inplace transform
*/
void fft(vector<base> &a, bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = (int)a.size();
    int data_size = n * sizeof(Complex_my);
    Complex_my *data_array = (Complex_my *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array[i].x = a[i].real();
        data_array[i].y = a[i].imag();
    }
    
    // Copying data to GPU
    Complex_my *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(dn, data_array, data_size, cudaMemcpyHostToDevice);
    // Bit reversal reordering
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);

    
    // Synchronize
    cudaDeviceSynchronize();
    // Iterative FFT with loop parallelism balancing
    for (int len = 2; len <= n; len <<= 1)
    {
        if (n / len > balance)
        {

            inplace_fft_outer<<<ceil((float)n / threads), threads>>>(A, len, n, threads, invert);
        }
        else
        {
            for (int i = 0; i < n; i += len)
            {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
            }
        }
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    // Copy data from GPU
    Complex_my *result;
    result = (Complex_my *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    for (int i = 0; i < n; i++)
    {
        a[i] = base(result[i].x, result[i].y);
    }
    // Free the memory blocks
    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return;
}

/**
* Performs 2D FFT 
* takes vector of complex vectors, invert and verbose as argument
* performs inplace FFT transform on input vector
*/
void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
{
    auto matrix = a;
    // Transform the rows
    if (verbose > 0)
        cout << "Transforming Rows" << endl;

    for (auto i = 0; i < matrix.size(); i++)
    {
        //cout<<i<<endl;
        fft(matrix[i], invert);
    }

    // preparing for transforming columns

    if (verbose > 0)
        cout << "Converting Rows to Columns" << endl;

    a = matrix;
    matrix.resize(a[0].size());
    for (int i = 0; i < matrix.size(); i++)
        matrix[i].resize(a.size());

    // Transposing matrix
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            matrix[j][i] = a[i][j];
        }
    }
    if (verbose > 0)
        cout << "Transforming Columns" << endl;

    // Transform the columns
    for (auto i = 0; i < matrix.size(); i++)
        fft(matrix[i], invert);

    if (verbose > 0)
        cout << "Storing the result" << endl;

    // Storing the result after transposing
    // [j][i] is getting value of [i][j]
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            a[j][i] = matrix[i][j];
        }
    }
}

/**
* Function to multiply two polynomial
* takes two polynomials represented as vectors as input
* return the product of two vectors
*/
vector<int> mult(vector<int> a, vector<int> b, int balance, int threads)
{
    // Creating complex vector from input vectors
    vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());

    // Padding with zero to make their size equal to power of 2
    size_t n = 1;
    while (n < max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;

    fa.resize(n), fb.resize(n);

    // Transforming both a and b
    // Converting to points form
    fft(fa, false, balance, threads), fft(fb, false, balance, threads);

    // performing point wise multipication of points
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];

    // Performing Inverse transform
    fft(fa, true, balance, threads);

    // Saving the real part as it will be the result
    vector<int> res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i)
        res[i] = int(fa[i].real() + 0.5);

    return res;
}

class FFT
{
public:
    /**
     * parallel FFT transform and inverse transform
     * Arguments vector of complex numbers, invert, balance, number of threads
     * Perform inplace transform
     */
    void fft(vector<base> &a, bool invert)
    {
        // Performing Bit reversal ordering
        int n = (int)a.size();

        for (int i = 1, j = 0; i < n; ++i)
        {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1)
                j -= bit;
            j += bit;
            if (i < j)
                swap(a[i], a[j]);
        }

        // Iteratinve FFT
        // This part of FFT is parallelizable
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * M_PI / len * (invert ? 1 : -1);
            base wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len)
            {
                base w(1);
                for (int j = 0; j < len / 2; ++j)
                {
                    base u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (invert)
            for (int i = 0; i < n; ++i)
                a[i] /= n;
        return;
    }

    /**
     * Performs 2D FFT 
     * takes vector of complex vectors, invert and verbose as argument
     * performs inplace FFT transform on input vector
     */
    void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
    {
        auto matrix = a;
        // Transform the rows
        if (verbose > 0)
            cout << "Transforming Rows" << endl;

        for (auto i = 0; i < matrix.size(); i++)
        {
            //cout<<i<<endl;
            fft(matrix[i], invert);
        }

        // preparing for transforming columns

        if (verbose > 0)
            cout << "Converting Rows to Columns" << endl;

        a = matrix;
        matrix.resize(a[0].size());
        for (int i = 0; i < matrix.size(); i++)
            matrix[i].resize(a.size());

        // Transposing matrix
        for (int i = 0; i < a.size(); i++)
        {
            for (int j = 0; j < a[0].size(); j++)
            {
                matrix[j][i] = a[i][j];
            }
        }
        if (verbose > 0)
            cout << "Transforming Columns" << endl;

        // Transform the columns
        for (auto i = 0; i < matrix.size(); i++)
            fft(matrix[i], invert);

        if (verbose > 0)
            cout << "Storing the result" << endl;

        // Storing the result after transposing
        // [j][i] is getting value of [i][j]
        for (int i = 0; i < a.size(); i++)
        {
            for (int j = 0; j < a[0].size(); j++)
            {
                a[j][i] = matrix[i][j];
            }
        }
    }

    /**
     * Function to multiply two polynomial
     * takes two polynomials represented as vectors as input
     * return the product of two vectors
     */
    vector<int> mult(vector<int> a, vector<int> b)
    {
        // Creating complex vector from input vectors
        vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());

        // Padding with zero to make their size equal to power of 2
        size_t n = 1;
        while (n < max(a.size(), b.size()))
            n <<= 1;
        n <<= 1;

        fa.resize(n), fb.resize(n);

        // Transforming both a and b
        // Converting to points form
        fft(fa, false), fft(fb, false);

        // performing point wise multipication of points
        for (size_t i = 0; i < n; ++i)
            fa[i] *= fb[i];

        // Performing Inverse transform
        fft(fa, true);

        // Saving the real part as it will be the result
        vector<int> res;
        res.resize(n);
        for (size_t i = 0; i < n; ++i)
            res[i] = int(fa[i].real() + 0.5);

        return res;
    }

    /**
     * Function to perform jpeg compression on image
     * takes image, threshold, verbose as input
     * image is represented as vector<vector>
     * perform inplace compression on the input
     */
    void compress_image(vector<vector<uint8_t>> &image, double threshold, int verbose = 1)
    {
        //Convert image to complex type

        vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
        for (auto i = 0; i < image.size(); i++)
        {
            for (auto j = 0; j < image[0].size(); j++)
            {
                complex_image[i][j] = image[i][j];
            }
        }
        if (verbose == 1)
        {
            cout << "input Image" << endl;
            //cout << image;
            cout << endl
                 << endl;
        }
        if (verbose > 1)
        {
            cout << "Complex Image" << endl;
            cout << complex_image;
            cout << endl
                 << endl;
        }

        //Perform 2D fft on image

        fft2D(complex_image, false, verbose);

        if (verbose == 1)
        {
            cout << "Performing FFT on Image" << endl;
            ///cout << complex_image;
            cout << endl
                 << endl;
        }

        //Threshold the fft

        // for (int i = 0; i < image_M.rows; ++i)
        //     for (int j = 0; j < image_M.cols; ++j)
        //         image_M.at<uint8_t>(i, j) = image[i][j];

        double maximum_value = 0.0;
        for (int i = 0; i < complex_image.size(); i++)
        {
            for (int j = 0; j < complex_image[0].size(); j++)
            {
                maximum_value = max(maximum_value, abs(complex_image[i][j]));
            }
        }
        threshold *= maximum_value;
        cout << "threshold :" << threshold << endl;
        int count = 0;

        // Setting values less than threshold to zero
        // This step is responsible for compression
        for (int i = 0; i < complex_image.size(); i++)
        {
            for (int j = 0; j < complex_image[0].size(); j++)
            {
                if (abs(complex_image[i][j]) < threshold)
                {
                    count++;
                    complex_image[i][j] = 0;
                }
            }
        }
        cout << count << endl;
        if (verbose > 1)
        {
            cout << "Thresholded Image" << endl;
            //cout << complex_image;
            cout << endl
                 << endl;
        }

        // Perform inverse FFT
        fft2D(complex_image, true, verbose);
        if (verbose > 1)
        {
            cout << "Inverted Image" << endl;
            //cout << complex_image;
            cout << endl
                 << endl;
        }
        //Convert to uint8 format
        // We will consider only the real part of the image
        for (int i = 0; i < complex_image.size(); i++)
        {
            for (int j = 0; j < complex_image[0].size(); j++)
            {
                image[i][j] = uint8_t(complex_image[i][j].real() + 0.5);
            }
        }
        if (verbose > 0)
        {
            cout << "Compressed Image" << endl;
            //cout << image;
        }
    }
};


#define N 1000
#define BALANCE 2

int main()
{
    vector<int> a = {1,1};
    vector<int> b = {1,2,3};
    auto multiplier = FFT();
    cout<<"A = "<<a;
    cout<<"B = "<<b;
    cout<<"A * B = "<<multiplier.mult(a, b)<<endl;
    /*
    std::vector<int> fa(N);
    std::generate(fa.begin(), fa.end(), std::rand);
    std::vector<int> fb(N);
    std::generate(fb.begin(), fb.end(), std::rand);
    freopen("out.txt", "w", stdout);
    for(int threads = 1; threads <= 1024; threads*=2){
	cerr << "For threads= " << threads << endl;
        /// For Parallel
        auto start = high_resolution_clock::now(); 

        auto result_parallel = mult(fa, fb, BALANCE, threads);

        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
      
        cout << threads << " " << duration.count();


        /// For Sequential
        auto multiplier = FFT();
        
        start = high_resolution_clock::now(); 
        auto result_sequential = multiplier.mult(fa, fb);

        stop = high_resolution_clock::now(); 
        duration = duration_cast<microseconds>(stop - start); 

        cout << " " << duration.count();

        cout << " " << (result_parallel == result_sequential) << endl;
        cout << endl;
    }
    */
    return 0;
}
