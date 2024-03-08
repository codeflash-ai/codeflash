def problem_p02757():
    def main():

        N, P = list(map(int, input().split()))

        S = [int(s) for s in list(eval(input()))][::-1]

        if P == 2 or P == 5:

            ans = 0

            for i in range(N):

                if S[i] % P == 0:

                    ans += N - i

            print(ans)

            return

        L = [0] * P

        L[0] = 1

        t = 0

        for i in range(N):

            t = (S[i] * pow(10, i, P) + t) % P

            L[t] += 1

        ans = 0

        for l in L:

            ans += l * (l - 1) // 2

        print(ans)

    if __name__ == "__main__":

        main()


problem_p02757()
