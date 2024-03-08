def problem_p02813():
    # abc150c

    from itertools import *

    n = int(eval(input()))

    p = tuple(map(int, input().split()))

    q = tuple(map(int, input().split()))

    l = list(permutations(list(range(1, n + 1))))

    print((abs(l.index(p) - l.index(q))))


problem_p02813()
