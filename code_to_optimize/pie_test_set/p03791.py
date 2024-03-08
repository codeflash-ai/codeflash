def problem_p03791():
    mod = 10**9 + 7

    N, *X = list(map(int, open(0).read().split()))

    stack = []

    ans = 1

    for x in X:

        stack.append(x)

        if x < 2 * len(stack) - 1:

            ans = (ans * len(stack)) % mod

            stack.pop()

        else:

            ans = (ans * len(stack)) % mod

    print(ans)


problem_p03791()
