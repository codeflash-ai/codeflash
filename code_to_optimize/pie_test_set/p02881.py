def problem_p02881(input_data):
    import numpy as np

    def main():

        N = int(eval(input_data))

        INF = 10**6 + 100

        x = np.arange(1, INF, dtype=np.int64)

        div = x[N % x == 0]

        # return (div)

        # return (N // div)

        ans = (div + N // div).min() - 2

        return ans

    if __name__ == "__main__":

        main()
