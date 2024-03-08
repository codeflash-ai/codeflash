def problem_p02720():
    def d_lunlun_number():

        import sys

        sys.setrecursionlimit(10**6)

        K = int(eval(input()))

        ans = []

        def dfs(num, current, digit_max):

            if current == digit_max:

                ans.append(num)

                return None

            for n in range(10):

                if abs((num % 10) - n) >= 2:

                    continue

                dfs(num * 10 + n, current + 1, digit_max)

        for leading in range(1, 10):

            for d in range(1, 11):

                dfs(leading, 1, d)

        return sorted(ans)[K - 1]

    print((d_lunlun_number()))


problem_p02720()
