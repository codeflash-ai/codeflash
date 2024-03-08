def problem_p03816():
    from collections import Counter

    N = int(eval(input()))

    A = list(map(int, input().split()))

    C = Counter(A)

    ans = len(C)

    if (N - ans) % 2 == 1:

        ans -= 1

    print(ans)


problem_p03816()
