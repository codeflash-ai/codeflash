def problem_p03208():
    N, K = list(map(int, input().split()))

    height = []

    for _ in range(N):

        height.append(int(eval(input())))

    height.sort()

    ans = float("inf")

    for i in range(N - K + 1):

        diff = height[i + K - 1] - height[i]

        ans = min(ans, diff)

    print(ans)


problem_p03208()
