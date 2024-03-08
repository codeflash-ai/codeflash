def problem_p03183():
    n = int(eval(input()))

    ans = 0

    p = [0] * 22222

    for w, s, v in sorted(
        [list(map(int, input().split())) for _ in [0] * n], key=lambda a: a[0] + a[1]
    ):

        for j in range(s, -1, -1):

            p[j + w] = max(p[j + w], p[j] + v)

            ans = max(ans, p[j + w])

    print(ans)


problem_p03183()
