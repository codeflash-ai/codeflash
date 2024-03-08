def problem_p02881():
    import numpy as np

    def main():

        N = int(eval(input()))

        INF = 10**6 + 100

        x = np.arange(1, INF, dtype=np.int64)

        div = x[N % x == 0]

        # print(div)

        # print(N // div)

        ans = (div + N // div).min() - 2

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02881()
