def problem_p02272():
    def merge_sort(a):

        return _merge_sort(a)

    def _merge_sort(a):

        if len(a) == 1:

            return a, 0

        m = len(a) / 2

        b, count1 = _merge_sort(a[0:m])

        c, count2 = _merge_sort(a[m : len(a)])

        ret, count3 = merge(b, c)

        return (ret, count1 + count2 + count3)

    def merge(b, c):

        a = []

        i = 0

        j = 0

        count = 0

        while i < len(b) and j < len(c):

            count += 1

            if b[i] > c[j]:

                a.append(c[j])

                j += 1

            else:

                a.append(b[i])

                i += 1

        if i == len(b):

            while j < len(c):

                count += 1

                a.append(c[j])

                j += 1

        elif j == len(c):

            while i < len(b):

                count += 1

                a.append(b[i])

                i += 1

        return a, count

    n = int(input())

    a = list(map(int, input().split(" ")))

    ret, count = merge_sort(a)

    print(" ".join(map(str, ret)))

    print(count)


problem_p02272()
