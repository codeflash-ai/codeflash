def problem_p03409():
    N = int(eval(input()))

    R = [(-1, -1)] * N

    for n in range(N):

        x, y = list(map(int, input().split()))

        R[n] = (x, y)

    B = [(-1, -1)] * N

    for n in range(N):

        x, y = list(map(int, input().split()))

        B[n] = (x, y)

    B = sorted(B)

    R = sorted(R, reverse=True)

    ans = 0

    for i, b in enumerate(B):

        tmp = []

        for j, r in enumerate(R):

            if r[0] == -1:

                continue

            if r[0] < b[0] and r[1] < b[1]:

                tmp.append((j, r))

        if tmp:

            tmp = sorted(tmp, key=lambda x: x[1][1], reverse=True)

            R[tmp[0][0]] = (-1, -1)

            ans += 1

    print(ans)


problem_p03409()
