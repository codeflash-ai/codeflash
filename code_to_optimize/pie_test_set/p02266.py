def problem_p02266():
    from collections import deque

    def scan(que, ps, way):

        Li = 0

        pp = ps

        while len(que) > 0:

            if way == 0:

                p = que.popleft()

            else:

                p = que.pop()

            if p < ps:

                Li += ps - p

            else:

                break

            pp = p

        return Li

    if __name__ == "__main__":

        D = eval(input())

        que = deque()

        p = 0

        for c in D:

            que.append(p)

            if c == "\\":
                p += -1

            elif c == "/":
                p += 1

        que.append(p)

        Ll = []

        Lr = []

        ls = que.popleft()

        if len(que) > 0:
            rs = que.pop()

        while len(que) > 0:

            while len(que) > 0:

                if que[0] < ls:

                    break

                else:

                    ls = que.popleft()

            while len(que) > 0:

                if que[-1] < rs:

                    break

                else:

                    rs = que.pop()

            if len(que) > 0:

                if ls < rs:
                    Ll.append(scan(que, ls, 0))

                else:
                    Lr.insert(0, scan(que, rs, 1))

        L = Ll + Lr

        if len(L) > 0:

            print((sum(L)))

            print((len(L), *L))

        else:

            print((0))

            print((0))


problem_p02266()
