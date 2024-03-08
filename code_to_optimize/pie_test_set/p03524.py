def problem_p03524():
    from collections import Counter

    def check(a, b):

        return abs(a - b) <= 1

    C = Counter(eval(input()))

    a, b, c = (C[s] for s in "abc")

    if check(a, b) and check(b, c) and check(c, a):

        print("YES")

    else:

        print("NO")


problem_p03524()
