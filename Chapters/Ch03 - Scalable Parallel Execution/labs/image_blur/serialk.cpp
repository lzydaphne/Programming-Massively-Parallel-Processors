#include <vector>
#include <iostream>
using namespace std;

class SerialKReduction
{
public:
    // Utility function to print a matrix
    static void printMatrix(const vector<vector<int>> &matrix, const string &name)
    {
        cout << name << ":\n";
        for (const auto &row : matrix)
        {
            for (int val : row)
            {
                cout << val << "\t";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    // Main demonstration function
    static void demonstrateSerialKReduction()
    {
        // Example: 2x6 matrix A multiplied by 6x2 matrix B
        // We'll split K (6) into chunks of size K' (2)
        vector<vector<int>> A = {
            {1, 2, 3, 4, 5, 6}, // M' = 2 rows
            {7, 8, 9, 10, 11, 12}};

        vector<vector<int>> B = {
            {1, 2}, // K = 6 rows
            {3, 4}, // N' = 2 columns
            {5, 6},
            {7, 8},
            {9, 10},
            {11, 12}};

        printMatrix(A, "Original Matrix A (2x6)");
        printMatrix(B, "Original Matrix B (6x2)");

        // K' = 2, so we'll have 3 chunks
        int M_prime = 2; // Number of rows in A
        int K = 6;       // Original K dimension
        int K_prime = 2; // Size of K chunks
        int N_prime = 2; // Number of columns in B

        // Initialize result matrix C_ij with zeros
        vector<vector<int>> C(M_prime, vector<int>(N_prime, 0));

        // Iterate over K chunks (serial-K reduction)
        for (int k = 0; k < K / K_prime; k++)
        {
            // Get current chunks of A and B
            vector<vector<int>> A_chunk = extractAChunk(A, k * K_prime, K_prime);
            vector<vector<int>> B_chunk = extractBChunk(B, k * K_prime, K_prime);

            cout << "\nProcessing chunk " << k + 1 << " of " << K / K_prime << ":\n";
            printMatrix(A_chunk, "A chunk");
            printMatrix(B_chunk, "B chunk");

            // Compute partial result for this chunk
            vector<vector<int>> partial = multiplyChunks(A_chunk, B_chunk);
            printMatrix(partial, "Partial result");

            // Accumulate (sum) into final result
            for (int i = 0; i < M_prime; i++)
            {
                for (int j = 0; j < N_prime; j++)
                {
                    C[i][j] += partial[i][j];
                }
            }
            printMatrix(C, "Accumulated result so far");
        }

        printMatrix(C, "Final Result Matrix C");
    }

private:
    // Extract a chunk from matrix A (columns k to k+k_prime)
    static vector<vector<int>> extractAChunk(
        const vector<vector<int>> &A,
        int start_col,
        int chunk_size)
    {
        vector<vector<int>> chunk(A.size(), vector<int>(chunk_size));
        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < chunk_size; j++)
            {
                chunk[i][j] = A[i][start_col + j];
            }
        }
        return chunk;
    }

    // Extract a chunk from matrix B (rows k to k+k_prime)
    static vector<vector<int>> extractBChunk(
        const vector<vector<int>> &B,
        int start_row,
        int chunk_size)
    {
        vector<vector<int>> chunk(chunk_size, vector<int>(B[0].size()));
        for (int i = 0; i < chunk_size; i++)
        {
            for (int j = 0; j < B[0].size(); j++)
            {
                chunk[i][j] = B[start_row + i][j];
            }
        }
        return chunk;
    }

    // Multiply chunks and return partial result
    static vector<vector<int>> multiplyChunks(
        const vector<vector<int>> &A_chunk,
        const vector<vector<int>> &B_chunk)
    {
        int M = A_chunk.size();
        int K = A_chunk[0].size();
        int N = B_chunk[0].size();

        vector<vector<int>> result(M, vector<int>(N, 0));

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < K; k++)
                {
                    result[i][j] += A_chunk[i][k] * B_chunk[k][j];
                }
            }
        }
        return result;
    }
};
// ... existing code ...

int main()
{
    SerialKReduction::demonstrateSerialKReduction();
    return 0;
}

// ... existing code ...