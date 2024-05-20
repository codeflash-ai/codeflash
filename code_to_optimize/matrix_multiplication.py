def matrix_multiply(A, B):
    """Multiply two matrices A and B using a slow triple-loop method."""
    if len(A[0]) != len(B):  # A's columns must be equal to B's rows
        raise ValueError("Matrices A and B cannot be multiplied")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # Perform the matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result
