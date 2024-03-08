def problem_p00711():
    import sys

    from collections import deque

    def solve():

        while 1:

            w, h = list(map(int, sys.stdin.readline().split()))

            if w == h == 0:

                return

            room = [[False] * w for i in range(h)]

            si = sj = -1

            for i in range(h):

                line = eval(input())

                for j, ch in enumerate(line):

                    if ch == "@":

                        room[i][j] = True

                        si = i

                        sj = j

                    elif ch == ".":

                        room[i][j] = True

            ans = bfs(w, h, room, si, sj)

            print(ans)

    dx = (
        1,
        0,
        -1,
        0,
    )

    dy = (
        0,
        1,
        0,
        -1,
    )

    def bfs(w, h, room, si, sj):

        visited = [[False] * w for i in range(h)]

        visited[si][sj] = True

        q = deque([(si, sj)])

        cnt = 1

        while q:

            ci, cj = q.popleft()

            for k in range(len(dx)):

                ni = ci + dy[k]

                nj = cj + dx[k]

                if 0 <= ni < h and 0 <= nj < w and room[ni][nj] and not visited[ni][nj]:

                    visited[ni][nj] = True

                    q.append((ni, nj))

                    cnt += 1

        return cnt

    if __name__ == "__main__":

        solve()


problem_p00711()
