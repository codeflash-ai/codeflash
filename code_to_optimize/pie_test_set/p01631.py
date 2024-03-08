def problem_p01631():
    def main():

        n = int(eval(input()))

        words = []

        scores = []

        for _ in range(n):

            word, score = input().split()

            words.append(word)

            scores.append(int(score))

        mp = ["#" * 6] + ["#" + eval(input()) + "#" for _ in range(4)] + ["#" * 6]

        t = int(eval(input()))

        def search(word):

            used = [[False] * 6 for _ in range(6)]

            vec = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))

            def _search(word, pos, x, y):

                if pos == len(word) - 1:
                    return 1

                used[y][x] = True

                ret = 0

                for dx, dy in vec:

                    nx, ny = x + dx, y + dy

                    if not used[ny][nx] and mp[ny][nx] == word[pos + 1]:

                        ret += _search(word, pos + 1, nx, ny)

                used[y][x] = False

                return ret

            ret = 0

            for y in range(1, 5):

                for x in range(1, 5):

                    if mp[y][x] == word[0]:
                        ret += _search(word, 0, x, y)

            return ret

        values = []

        weights = []

        for word, score in zip(words, scores):

            cnt = search(word)

            acc = 1

            while cnt >= acc:

                cnt -= acc

                values.append(score * acc)

                weights.append(len(word) * acc)

                acc *= 2

            if cnt:

                values.append(score * cnt)

                weights.append(len(word) * cnt)

        dp = [0] * (t + 1)

        for v, w in zip(values, weights):

            for x in range(max(-1, t - w), -1, -1):

                dp[x + w] = max(dp[x + w], dp[x] + v)

        print((max(dp)))

    main()


problem_p01631()
