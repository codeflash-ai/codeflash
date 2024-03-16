def problem_p03524(input_data):
    from collections import Counter

    def check(a, b):

        return abs(a - b) <= 1

    C = Counter(eval(input_data))

    a, b, c = (C[s] for s in "abc")

    if check(a, b) and check(b, c) and check(c, a):

        return "YES"

    else:

        return "NO"
