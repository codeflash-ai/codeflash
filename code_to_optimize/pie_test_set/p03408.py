def problem_p03408():
    n = int(eval(input()))

    S = [str(eval(input())) for _ in range(n)]

    m = int(eval(input()))

    T = [str(eval(input())) for _ in range(m)]

    from collections import Counter

    CS = Counter(S)

    CT = Counter(T)

    ans = 0

    for k in list(CS.keys()):

        if k not in CT:

            ans = max(CS[k], ans)

        else:

            ans = max(ans, max(0, CS[k] - CT[k]))

    print(ans)


problem_p03408()
