def problem_p03569(input_data):
    import sys

    sys.setrecursionlimit(10**7)

    f_inf = float("inf")

    mod = 10**9 + 7

    def resolve():

        s = eval(input_data)

        ss = []

        for i in s:

            if i != "x":

                ss.append(i)

        if ss != ss[::-1]:

            return -1

        else:

            n = len(ss) + 1

            cnt = 0

            cnt_x = [0] * (n)

            pos = 0

            for i in range(len(s)):

                if s[i] == "x":

                    cnt += 1

                if s[i] != "x" or i == len(s) - 1:

                    cnt_x[pos] = cnt

                    cnt = 0

                    pos += 1

            res = 0

            for i in range(n // 2):

                res += abs(cnt_x[i] - cnt_x[-(i + 1)])

            return res

    if __name__ == "__main__":

        resolve()
