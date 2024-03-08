def problem_p03785():
    def main() -> None:

        n, c, k = list(map(int, input().split()))

        t = [int(eval(input())) for _ in range(n)]

        ans = wait_start = wait_people = 0

        for ti in sorted(t):

            if ti - wait_start > k or wait_people == c:

                wait_people = 0

            wait_people += 1

            if wait_people == 1:

                wait_start = ti

                ans += 1

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03785()
