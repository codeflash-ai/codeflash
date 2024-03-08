def problem_p00748():
    def main():

        INIT = 100

        query = []

        ans = []

        while True:

            q = int(eval(input()))

            if q == 0:

                break

            query.append(q)

        MAX = max(query)

        table = [INIT] * (MAX + 1)

        table[0] = 0

        all_item = [i * (i + 1) * (i + 2) // 6 for i in range(1, 181)]

        odd_item = [i for i in all_item if i % 2]

        eve_item = [i for i in all_item if not i % 2]

        for v in odd_item:

            for j in range(v, MAX + 1):

                tjv = table[j - v]

                if table[j] > tjv + 1:

                    table[j] = tjv + 1

        for q in query:

            ans.append(table[q])

        for v in eve_item:

            for j in range(v, MAX + 1):

                tjv = table[j - v]

                if tjv < 5 and table[j] > tjv + 1:

                    table[j] = tjv + 1

        for i, q in enumerate(query):

            print((table[q], ans[i]))

    main()


problem_p00748()
