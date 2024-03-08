def problem_p03007():
    N = int(eval(input()))

    A = [int(a) for a in input().split()]

    B = [abs(a) for a in A]

    for i in range(N):

        if A[i] > 0:

            po = i

            break

    else:

        po = -1

    for i in range(N):

        if A[i] < 0:

            ne = i

            break

    else:

        ne = -1

    mi = min(B)

    for i in range(N):

        if B[i] == mi:

            mii = i

            break

    else:

        mii = -1

    if po >= 0 and ne >= 0:

        print((sum(B)))

        t1, t2 = A[po], A[ne]

        if po > ne:

            po, ne = ne, po

        # print("pone =", po, ne)

        if po == ne:

            del A[ne]

            A = [0] + A

        else:

            del A[ne]

            del A[po]

            A = [t1, t2] + A

    else:

        print((sum(B) - 2 * mi))

        t = A[mii]

        del A[mii]

        A = [t] + A

    # print(A)

    prev = A[0]

    for i in range(1, N):

        a, b = max(prev, A[i]), min(prev, A[i])

        if i == N - 1 or A[i + 1] < 0:

            print((a, b))

            prev = a - b

        else:

            print((b, a))

            prev = b - a


problem_p03007()
