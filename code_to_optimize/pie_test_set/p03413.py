def problem_p03413():
    import numpy as np

    def main():

        n = int(input())

        a = [int(x) for x in input().split()]

        best, solution = int(-1e10), []

        for p in range(2):

            c = a.copy()

            indexes = list(range(p, n, 2))

            if len(indexes) == 0:

                continue

            maks = max([c[i] for i in indexes])

            pick = next(filter(lambda i: c[i] == maks, indexes))

            chosen = set([pick] if maks <= 0 else filter(lambda i: c[i] > 0, indexes))

            is_chosen = [(i in chosen) for i in range(n)]

            tot, res = sum([c[i] for i in chosen]), []

            for i in reversed(range(n)):

                if not is_chosen[i]:

                    if i == 0 or i + 1 == len(c):

                        res += [i]

                        del c[i], is_chosen[i]

                    elif is_chosen[i - 1] == is_chosen[i + 1]:

                        res += [i]

                        c[i - 1] += c[i + 1]

                        del c[i + 1], c[i], is_chosen[i + 1], is_chosen[i]

            if len(c) > 1 and not is_chosen[0]:

                res += [0]

                del c[0], is_chosen[0]

            if tot > best:

                best, solution = tot, np.array(res) + 1

        print(best, len(solution), *solution, sep="\n")

    if __name__ == "__main__":

        main()


problem_p03413()
