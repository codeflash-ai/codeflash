def problem_p00769():
    def solve(a):

        if type(a[0]) is int:

            a = [a[i] / 2 + 1 for i in range(len(a))]

            return sum(sorted(a)[: len(a) / 2 + 1])

        else:

            return sum(sorted(solve(a[i]) for i in range(len(a)))[: len(a) / 2 + 1])

    for i in range(eval(input())):

        A = eval(input().replace("][", "],["))

        print(solve(A))


problem_p00769()
