def problem_p02248():
    def kmpTable(w):

        lw = len(w)

        nx = [-1] * (lw + 1)

        j = -1

        for i in range(lw):

            while j >= 0 and w[i] != w[j]:

                j = nx[j]

            j += 1

            nx[i + 1] = j

        return nx

    def kmpSearch(s, w):

        ls = len(s)

        start = 0

        w_idx = 0

        ret = []

        nx = kmpTable(w)

        while start + w_idx < ls:

            if s[start + w_idx] == w[w_idx]:

                w_idx += 1

                if w_idx == len(w):

                    ret.append(start)

                    start = start + w_idx - nx[w_idx]

                    w_idx = nx[w_idx]

            else:

                if w_idx == 0:

                    start += 1

                else:

                    start = start + w_idx - nx[w_idx]

                    w_idx = nx[w_idx]

        return ret

    T = eval(input())

    P = eval(input())

    ans = kmpSearch(T, P)

    if ans:

        print(("\n".join(map(str, ans))))


problem_p02248()
