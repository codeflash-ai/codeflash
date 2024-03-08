def problem_p03252():
    def main():

        import collections

        S = eval(input())

        T = eval(input())

        table = collections.defaultdict(set)

        table2 = collections.defaultdict(set)

        a = []

        b = []

        for i, xy in enumerate(zip(S, T)):

            x, y = xy

            if x not in table:

                a.append(i)

            if y not in table2:

                b.append(i)

            table[x].add(i)

            table2[y].add(i)

        flag = True

        if a != b:

            flag = False

        for i in a:

            if flag:

                if table[S[i]] == table2[T[i]]:

                    continue

                else:

                    flag = False

                    break

        if flag:

            print("Yes")

        else:

            print("No")

    main()


problem_p03252()
