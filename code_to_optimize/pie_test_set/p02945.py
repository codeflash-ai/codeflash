def problem_p02945():
    import numpy as np

    a, b = list(map(int, input().split()))

    ans = [a + b, a - b, a * b]

    print((np.amax(ans)))


problem_p02945()
