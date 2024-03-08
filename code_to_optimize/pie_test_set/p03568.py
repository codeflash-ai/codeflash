def problem_p03568():
    from functools import reduce

    from itertools import product

    N = int(eval(input()))

    (*A,) = list(map(int, input().split()))

    ans = 0

    for t in product([-1, 0, 1], repeat=N):

        prod = reduce(lambda x, y: x * y, [a + t[i] for i, a in enumerate(A)])

        if prod % 2 == 0:
            ans += 1

    print(ans)


problem_p03568()
