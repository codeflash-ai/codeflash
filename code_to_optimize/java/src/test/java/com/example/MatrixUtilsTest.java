package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixUtilsTest {

    @Test
    void testMultiply() {
        int[][] a = {{1, 2}, {3, 4}};
        int[][] b = {{5, 6}, {7, 8}};
        int[][] result = MatrixUtils.multiply(a, b);

        assertEquals(19, result[0][0]);
        assertEquals(22, result[0][1]);
        assertEquals(43, result[1][0]);
        assertEquals(50, result[1][1]);
    }

    @Test
    void testTranspose() {
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        int[][] result = MatrixUtils.transpose(matrix);

        assertEquals(3, result.length);
        assertEquals(2, result[0].length);
        assertEquals(1, result[0][0]);
        assertEquals(4, result[0][1]);
    }

    @Test
    void testAdd() {
        int[][] a = {{1, 2}, {3, 4}};
        int[][] b = {{5, 6}, {7, 8}};
        int[][] result = MatrixUtils.add(a, b);

        assertEquals(6, result[0][0]);
        assertEquals(8, result[0][1]);
        assertEquals(10, result[1][0]);
        assertEquals(12, result[1][1]);
    }

    @Test
    void testScalarMultiply() {
        int[][] matrix = {{1, 2}, {3, 4}};
        int[][] result = MatrixUtils.scalarMultiply(matrix, 3);

        assertEquals(3, result[0][0]);
        assertEquals(6, result[0][1]);
        assertEquals(9, result[1][0]);
        assertEquals(12, result[1][1]);
    }

    @Test
    void testDeterminant() {
        assertEquals(1, MatrixUtils.determinant(new int[][]{{1}}));
        assertEquals(-2, MatrixUtils.determinant(new int[][]{{1, 2}, {3, 4}}));
        assertEquals(0, MatrixUtils.determinant(new int[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
    }

    @Test
    void testRotate90Clockwise() {
        int[][] matrix = {{1, 2}, {3, 4}};
        int[][] result = MatrixUtils.rotate90Clockwise(matrix);

        assertEquals(3, result[0][0]);
        assertEquals(1, result[0][1]);
        assertEquals(4, result[1][0]);
        assertEquals(2, result[1][1]);
    }

    @Test
    void testIsSymmetric() {
        assertTrue(MatrixUtils.isSymmetric(new int[][]{{1, 2}, {2, 1}}));
        assertFalse(MatrixUtils.isSymmetric(new int[][]{{1, 2}, {3, 4}}));
    }

    @Test
    void testRowWithMaxSum() {
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}, {1, 1, 1}};
        assertEquals(1, MatrixUtils.rowWithMaxSum(matrix));
    }

    @Test
    void testSearchElement() {
        int[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        int[] result = MatrixUtils.searchElement(matrix, 5);

        assertNotNull(result);
        assertEquals(1, result[0]);
        assertEquals(1, result[1]);

        assertNull(MatrixUtils.searchElement(matrix, 10));
    }

    @Test
    void testTrace() {
        assertEquals(5, MatrixUtils.trace(new int[][]{{1, 2}, {3, 4}}));
        assertEquals(15, MatrixUtils.trace(new int[][]{{1, 0, 0}, {0, 5, 0}, {0, 0, 9}}));
    }

    @Test
    void testIdentity() {
        int[][] result = MatrixUtils.identity(3);

        assertEquals(1, result[0][0]);
        assertEquals(0, result[0][1]);
        assertEquals(1, result[1][1]);
        assertEquals(1, result[2][2]);
    }

    @Test
    void testPower() {
        int[][] matrix = {{1, 1}, {1, 0}};
        int[][] result = MatrixUtils.power(matrix, 3);

        assertEquals(3, result[0][0]);
        assertEquals(2, result[0][1]);
    }
}
