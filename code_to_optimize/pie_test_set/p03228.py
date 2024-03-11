def problem_p03228(input_data):
    A, B, K = list(map(int, input_data.split()))

    for k in range(K):

        if k % 2 == 0:

            if A % 2 == 0:

                A //= 2

                B += A

            else:

                A -= 1

                A //= 2

                B += A

        else:

            if B % 2 == 0:

                B //= 2

                A += B

            else:

                B -= 1

                B //= 2

                A += B

    return (A, B)
