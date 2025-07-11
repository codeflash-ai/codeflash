def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrices A and B cannot be multiplied")

    num_rows_A = len(A)
    num_cols_A = len(A[0])
    num_cols_B = len(B[0])

    # Transpose B for better cache locality in inner loop
    B_T = [list(col) for col in zip(*B)]

    result = [[0] * num_cols_B for _ in range(num_rows_A)]

    range_num_rows_A = range(num_rows_A)
    range_num_cols_B = range(num_cols_B)
    range_num_cols_A = range(num_cols_A)

    for i in range_num_rows_A:
        rowA = A[i]
        res_row = result[i]
        for j in range_num_cols_B:
            rowB = B_T[j]
            # Compute dot product directly
            total = 0
            for k in range_num_cols_A:
                total += rowA[k] * rowB[k]
            res_row[j] = total
    return result
