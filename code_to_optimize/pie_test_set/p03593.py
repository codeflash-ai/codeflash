def problem_p03593():
    H, W = list(map(int, input().split()))

    C = {}

    for _ in range(H):

        A = input().strip()

        for i in range(W):

            if A[i] not in C:

                C[A[i]] = 0

            C[A[i]] += 1

    cnt4 = 0

    cnt1 = 0

    for a in C:

        if C[a] >= 4:

            cnt4 += C[a] // 4

            C[a] = C[a] % 4

        if C[a] % 2 == 1:

            cnt1 += 1

    if cnt4 < (W // 2) * (H // 2):

        flag = 0

    else:

        if H % 2 == 1 and W % 2 == 1:

            if cnt1 == 1:

                flag = 1

            else:

                flag = 0

        else:

            if cnt1 == 0:

                flag = 1

            else:

                flag = 0

    if flag == 1:

        print("Yes")

    else:

        print("No")


problem_p03593()
