def problem_p03427():
    def solve(N):

        if N < 10:

            return N

        num = 9

        while num < N:

            num *= 10

            num += 9

        num -= 9

        num //= 10

        k = int(str(N)[0] + str(num))

        if k <= N:

            return sum(int(c) for c in str(k))

        else:

            k = int(str(N)[0] + str(num)) - num - 1

            return sum(int(c) for c in str(k))

    if __name__ == "__main__":

        N = int(eval(input()))

        print((solve(N)))


problem_p03427()
