def problem_p02274():
    import sys

    cnt = 0

    def merge_sort(a, left, right):

        if left + 1 < right:

            mid = int(left + (right - left) / 2)

            merge_sort(a, left, mid)

            merge_sort(a, mid, right)

            merge(a, left, mid, right)

    def merge(a, left, mid, right):

        n1 = mid - left

        n2 = right - mid

        L = a[left : left + n1]

        R = a[mid : mid + n2]

        L.append(sys.maxsize)

        R.append(sys.maxsize)

        i = j = 0

        tmp = n1

        for k in range(left, right):

            if L[i] <= R[j]:

                if R[j] != sys.maxsize:

                    tmp -= 1

                a[k] = L[i]

                i += 1

            else:

                if L[i] != sys.maxsize:

                    global cnt

                    cnt += tmp

                a[k] = R[j]

                j += 1

    n = int(eval(input()))

    a = list(map(int, input().split()))

    merge_sort(a, 0, len(a))

    print(cnt)


problem_p02274()
