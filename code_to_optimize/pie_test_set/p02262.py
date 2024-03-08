def problem_p02262():
    import sys

    if sys.version_info[0] >= 3:
        raw_input = input

    def insertionSort(a, g):

        global cnt

        for i in range(g, len(a)):

            v = a[i]

            j = i - g

            while j >= 0 and a[j] > v:

                a[j + g] = a[j]

                j = j - g

                cnt += 1

                a[j + g] = v

    def shellSort(a):

        global cnt

        cnt = 0

        g = []

        h = 1

        while h <= len(a):

            g.append(h)

            h = 3 * h + 1

        g.reverse()

        m = len(g)

        print(m)

        print((" ".join(map(str, g))))

        for i in range(m):

            insertionSort(a, g[i])

    a = [int(input()) for i in range(int(input()))]

    shellSort(a)

    print(cnt)

    for e in a:
        print(e)


problem_p02262()
