def problem_p02867():
    import numpy as np

    N = int(eval(input()))

    A = np.array(list(map(int, input().split())))

    B = np.array(list(map(int, input().split())))

    Bsi = np.argsort(B)

    A2 = A[Bsi]

    Bs = B[Bsi]

    A2si = np.argsort(A2)

    As = A2[A2si]

    C = Bs - As

    if any(C[C < 0]):

        ans = "No"

    else:

        D = A2 - As

        if any(D == 0):

            ans = "Yes"

        elif all(As[1::] - Bs[:-1:] > 0):

            pi = 0

            i = 0

            while i < N - 1:

                pi = A2si[pi]

                if pi == 0:

                    ans = "Yes"

                    break

                else:

                    i += 1

            else:

                ans = "No"

        else:

            ans = "Yes"

    print(ans)


problem_p02867()
