def problem_p03262():
    def gcd(a, b):

        while b:

            a, b = b, a % b

        return a

    def gcd_n(numbers):

        ret = numbers[0]

        for i in range(len(numbers)):

            ret = gcd(ret, numbers[i])

        return ret

    def main():

        N, X = list(map(int, input().split()))

        x = [abs(X - int(i)) for i in input().split()]

        print((gcd_n(x)))

        return

    main()


problem_p03262()
