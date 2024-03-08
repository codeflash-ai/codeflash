def problem_p03048():
    r, g, b, n = list(map(int, input().split()))

    ans = 0

    r, g, b = max(r, g, b), r + g + b - max(r, g, b) - min(r, g, b), min(r, g, b)

    for i in range(n // r + 1):

        for j in range(n // g + 1):

            if (n - i * r - j * g) % b or (n - i * r - j * g) < 0:

                continue

            ans += 1

    print(ans)


problem_p03048()
