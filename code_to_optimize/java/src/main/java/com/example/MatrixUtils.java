package com.example;

/**
 * Matrix operations.
 */
public class MatrixUtils {

    /**
     * Multiply two matrices.
     *
     * @param a First matrix
     * @param b Second matrix
     * @return Product matrix
     */
    public static int[][] multiply(int[][] a, int[][] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) {
            return new int[0][0];
        }

        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;

        if (colsA != b.length) {
            throw new IllegalArgumentException("Matrix dimensions don't match");
        }

        int[][] result = new int[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                int sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum = sum + a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    /**
     * Transpose a matrix.
     *
     * @param matrix Input matrix
     * @return Transposed matrix
     */
    public static int[][] transpose(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new int[0][0];
        }

        int rows = matrix.length;
        int cols = matrix[0].length;

        int[][] result = new int[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

    /**
     * Add two matrices element by element.
     *
     * @param a First matrix
     * @param b Second matrix
     * @return Sum matrix
     */
    public static int[][] add(int[][] a, int[][] b) {
        if (a == null || b == null) {
            return new int[0][0];
        }

        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Matrix dimensions must match");
        }

        int rows = a.length;
        int cols = a[0].length;

        int[][] result = new int[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    /**
     * Multiply matrix by scalar.
     *
     * @param matrix Input matrix
     * @param scalar Scalar value
     * @return Scaled matrix
     */
    public static int[][] scalarMultiply(int[][] matrix, int scalar) {
        if (matrix == null || matrix.length == 0) {
            return new int[0][0];
        }

        int rows = matrix.length;
        int cols = matrix[0].length;

        int[][] result = new int[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }

        return result;
    }

    /**
     * Calculate determinant using recursive expansion.
     *
     * @param matrix Square matrix
     * @return Determinant value
     */
    public static long determinant(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }

        int n = matrix.length;

        if (n == 1) {
            return matrix[0][0];
        }

        if (n == 2) {
            return (long) matrix[0][0] * matrix[1][1] - (long) matrix[0][1] * matrix[1][0];
        }

        long det = 0;
        for (int j = 0; j < n; j++) {
            int[][] subMatrix = new int[n - 1][n - 1];

            for (int row = 1; row < n; row++) {
                int subCol = 0;
                for (int col = 0; col < n; col++) {
                    if (col != j) {
                        subMatrix[row - 1][subCol] = matrix[row][col];
                        subCol++;
                    }
                }
            }

            int sign = (j % 2 == 0) ? 1 : -1;
            det = det + sign * matrix[0][j] * determinant(subMatrix);
        }

        return det;
    }

    /**
     * Rotate matrix 90 degrees clockwise.
     *
     * @param matrix Input matrix
     * @return Rotated matrix
     */
    public static int[][] rotate90Clockwise(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new int[0][0];
        }

        int rows = matrix.length;
        int cols = matrix[0].length;

        int[][] result = new int[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][rows - 1 - i] = matrix[i][j];
            }
        }

        return result;
    }

    /**
     * Check if matrix is symmetric.
     *
     * @param matrix Input matrix
     * @return true if symmetric
     */
    public static boolean isSymmetric(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return true;
        }

        int n = matrix.length;

        if (n != matrix[0].length) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] != matrix[j][i]) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Find row with maximum sum.
     *
     * @param matrix Input matrix
     * @return Index of row with maximum sum
     */
    public static int rowWithMaxSum(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return -1;
        }

        int maxRow = 0;
        int maxSum = Integer.MIN_VALUE;

        for (int i = 0; i < matrix.length; i++) {
            int sum = 0;
            for (int j = 0; j < matrix[i].length; j++) {
                sum = sum + matrix[i][j];
            }
            if (sum > maxSum) {
                maxSum = sum;
                maxRow = i;
            }
        }

        return maxRow;
    }

    /**
     * Search for element in matrix.
     *
     * @param matrix Input matrix
     * @param target Value to find
     * @return Array [row, col] or null if not found
     */
    public static int[] searchElement(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return null;
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] == target) {
                    return new int[]{i, j};
                }
            }
        }

        return null;
    }

    /**
     * Calculate trace (sum of diagonal elements).
     *
     * @param matrix Square matrix
     * @return Trace value
     */
    public static int trace(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }

        int sum = 0;
        int n = Math.min(matrix.length, matrix[0].length);

        for (int i = 0; i < n; i++) {
            sum = sum + matrix[i][i];
        }

        return sum;
    }

    /**
     * Create identity matrix of given size.
     *
     * @param n Size of matrix
     * @return Identity matrix
     */
    public static int[][] identity(int n) {
        if (n <= 0) {
            return new int[0][0];
        }

        int[][] result = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    result[i][j] = 1;
                } else {
                    result[i][j] = 0;
                }
            }
        }

        return result;
    }

    /**
     * Raise matrix to a power using repeated multiplication.
     *
     * @param matrix Square matrix
     * @param power Exponent
     * @return Matrix raised to power
     */
    public static int[][] power(int[][] matrix, int power) {
        if (matrix == null || matrix.length == 0 || power < 0) {
            return new int[0][0];
        }

        int n = matrix.length;

        if (power == 0) {
            return identity(n);
        }

        int[][] result = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = matrix[i][j];
            }
        }

        for (int p = 1; p < power; p++) {
            result = multiply(result, matrix);
        }

        return result;
    }
}
