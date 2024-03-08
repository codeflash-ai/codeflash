def problem_p02962():
    s = eval(input())

    t = eval(input())

    def z_algorithm(s):

        n = len(s)

        res = [0] * n

        i = 1

        j = 0

        while i < n:

            # i番目以降の一致文字数

            while i + j < n and s[j] == s[i + j]:

                j += 1

            res[i] = j

            # 一文字も一致しない場合，次の文字へ

            if j == 0:

                i += 1

                continue

            # 一致したところまでを埋める

            k = 1

            while i + k < n and k + res[k] < j:

                res[i + k] = res[k]

                k += 1

            i += k

            j -= k

        return res

    n = len(t)

    if n > len(s):

        ss = s * (n // len(s) + 1)

    else:

        ss = s * 2

    z = z_algorithm(t + ss * 2)

    z = [min(z[i], n) for i in range(n, len(z))]

    ans = 0

    for i in range(n):

        c = 0

        for v in z[i::n]:

            if v == n:

                c += 1

            else:

                c = 0

            ans = max(ans, c)

    if ans <= len(ss) // n:

        print(ans)

    else:

        print((-1))


problem_p02962()
