def problem_p03382():
    import numpy as np

    eval(input())
    a = list(map(int, input().split(" ")))
    m = max(a)
    a.remove(m)
    b = [abs(2 * i - m) for i in a]
    j = b.index(min(b))
    print((m, a[j]))


problem_p03382()
