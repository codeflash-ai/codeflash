def problem_p02635(input_data):
    def main():

        mod = 998244353

        s, k = input_data.split()

        k = int(k)

        n = len(s)

        one = s.count("1")

        cnt = 0

        zero_list = []

        for i in range(n):

            if s[i] == "0":

                zero_list.append(cnt)

                cnt = 0

            else:

                cnt += 1

        z = 0

        mm = min(one, k)

        dp = [[0] * (one + 1) for _ in [0] * (one + 1)]

        dp[0][0] = 1

        for i in range(len(zero_list)):

            dp2 = [[0] * (mm + 1) for _ in [0] * (one + 1)]

            base = zero_list[i]

            # j:何個今までに入れたか

            for j in range(one + 1):

                # l:何個入れるか

                for l in range(one + 1 - j):

                    if l < z + base - j:

                        continue

                    ml = max(l - base, 0)

                    # p:これまでのペナルティ

                    for p in range(min(one, k) + 1):

                        q = p + ml

                        if q <= mm:

                            dp2[j + l][q] = (dp2[j + l][q] + dp[j][p]) % mod

                        else:

                            break

            z += base

            dp = dp2

        return sum([sum(i) for i in dp]) % mod

    main()
