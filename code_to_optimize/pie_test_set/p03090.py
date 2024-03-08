def problem_p03090():
    N = int(eval(input()))

    ban = N if N % 2 else N + 1

    ans = []

    for i in range(1, N):

        for j in range(i + 1, N + 1):

            if i + j == ban:
                continue

            ans.append((i, j))

    print((len(ans)))

    for a, b in ans:

        print((a, b))


problem_p03090()
