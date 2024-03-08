def problem_p02670():
    import numba

    import numpy as np

    N = int(eval(input()))

    P = np.array([int(c) - 1 for c in input().split()])

    # from random import shuffle

    # N = 500

    # P = list(range(N*N))

    # shuffle(P)

    # P = np.array(P)

    @numba.njit("i8(i8,i8[:])", cache=True)
    def proc(N, P):

        ans = 0

        Removed = np.zeros((500, 500), dtype=np.int8)

        Mat = np.zeros((500, 500), dtype=np.int16)

        Dy, Dx = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])

        for y in range(N):

            for x in range(N):

                Mat[y, x] = min(y, N - 1 - y, x, N - 1 - x)

        # print(Mat)

        Buf = [(0, 0) for _ in range(1 << 17)]

        Buf_idx = 0

        for i in range(N * N):

            p = P[i]

            y, x = divmod(p, N)

            ans += Mat[y, x]

            Removed[y, x] = 1

            Buf[Buf_idx] = (y, x)

            Buf_idx += 1

            while Buf_idx:

                Buf_idx -= 1

                vy, vx = Buf[Buf_idx]

                vr = Removed[vy, vx]

                val = Mat[vy, vx]

                # for dy, dx in zip(Dy, Dx):

                for j in range(4):

                    uy, ux = vy + Dy[j], vx + Dx[j]

                    if not (0 <= uy < N and 0 <= ux < N):

                        continue

                    ur = Removed[uy, ux]

                    if vr:

                        if Mat[uy, ux] >= val + 1:

                            Mat[uy, ux] = val

                            Buf[Buf_idx] = (uy, ux)

                            Buf_idx += 1

                    else:

                        if Mat[uy, ux] >= val + 2:

                            Mat[uy, ux] = val + 1

                            Buf[Buf_idx] = (uy, ux)

                            Buf_idx += 1

        # for m in Mat:

        #     print(m)

        return ans

    print((proc(N, P)))


problem_p02670()
