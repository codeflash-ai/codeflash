def problem_p03325():
    import numpy as np

    N = int(eval(input()))

    a = list(map(int, input().split()))

    ans = 0

    a = np.array(a)

    while a.size > 0:

        a = a[a % 2 == 0]

        ans += len(a)

        a = a // 2

    print(ans)


problem_p03325()
