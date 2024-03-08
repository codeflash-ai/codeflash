def problem_p03428():
    from collections import Counter

    import numpy as np

    def part(pts, a, ans):

        b = np.dot(pts, [np.sin(a), np.cos(a)])

        c = np.argmax(b, axis=0)

        for i, v in list(Counter(c).items()):

            ans[i] += v

    def solve(n, pts):

        ans = [0] * n

        part(pts, np.linspace(0, np.pi, 100000, endpoint=False), ans)

        part(pts, np.linspace(np.pi, 2 * np.pi, 100000, endpoint=False), ans)

        return [v / 200000 for v in ans]

    n = int(eval(input()))

    pts = np.array([list(map(int, input().split())) for _ in range(n)])

    print(("\n".join(map(str, solve(n, pts)))))


problem_p03428()
