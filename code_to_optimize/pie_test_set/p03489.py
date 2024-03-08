def problem_p03489():
    # ARC087C - Good Sequence (ABC082C)

    import sys

    input = sys.stdin.readline

    def main():

        n = int(eval(input()))

        lst = list(map(int, input().rstrip().split()))

        cnt = {}

        for i in lst:

            if i not in cnt:

                cnt[i] = 0

            cnt[i] += 1

        ans = 0

        for i, j in list(cnt.items()):

            if i > j:

                ans += j

            elif i < j:

                ans += j - i

        print(ans)

    if __name__ == "__main__":

        main()


problem_p03489()
