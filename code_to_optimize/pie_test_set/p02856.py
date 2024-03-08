def problem_p02856():

    def resolve():

        N = int(eval(input()))

        S = 0

        D = 0

        for i in range(N):

            d, c = list(map(int, input().split()))

            S += d * c

            D += c

        print(((D - 1) + (S - 1) // 9))

    if __name__ == "__main__":

        resolve()


problem_p02856()
