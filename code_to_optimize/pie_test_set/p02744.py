def problem_p02744():
    def main():

        from string import ascii_lowercase

        dic = {i: s for i, s in enumerate(ascii_lowercase)}

        N = int(input())

        ans = []

        def dfs(s, mx):

            if len(s) == N:

                ans.append(s)

                return

            else:

                for i in range(mx + 2):

                    mx = max(mx, i)

                    dfs(s + dic[i], mx)

        dfs("a", 0)

        print(*ans, sep="\n")

    if __name__ == "__main__":

        main()


problem_p02744()
