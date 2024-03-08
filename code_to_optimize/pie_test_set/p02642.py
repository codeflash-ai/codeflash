def problem_p02642():
    import sys

    read = sys.stdin.read

    readlines = sys.stdin.readlines

    import numpy as np

    def main():

        a = np.array(read().split(), np.int32)[1:]

        count = np.zeros(10**6 + 1, np.int32)

        for ae in a:

            if count[ae] <= 1:

                count[::ae] += 1

        r = 0

        for ae in a:

            r += count[ae] == 1

        print(r)

    if __name__ == "__main__":

        main()


problem_p02642()
