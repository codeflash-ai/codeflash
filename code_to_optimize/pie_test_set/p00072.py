def problem_p00072():
    # http://ja.wikipedia.org/wiki/%E3%83%97%E3%83%AA%E3%83%A0%E6%B3%95

    class vertex:

        def __init__(self):

            self.is_in_V = False

    def is_single_in_V(v, p):

        a = v[p[0]].is_in_V

        b = v[p[1]].is_in_V

        return (a and not b) or (not a and b)

    def all_in_V(v):

        for e in v:

            if not e.is_in_V:

                return False

        return True

    def add_in_V(v, d):

        v[d[0]].is_in_V = True

        v[d[1]].is_in_V = True

    def remove_all_in_V_pass(v, d):

        for i in range(len(d)):

            try:

                if v[d[i][0]].is_in_V and v[d[i][1]].is_in_V:

                    d.pop(i)

            except:

                pass

    def solve(nu, mu, du):

        E = []

        vertexes = [vertex() for i in range(nu)]

        vertexes[0].is_in_V = True

        while True:

            if all_in_V(vertexes):

                break

            min_dist, ind = float("inf"), float("inf")

            for i in range(len(du)):

                if is_single_in_V(vertexes, du[i][0:2]):

                    if min_dist > du[i][2]:

                        min_dist, ind = du[i][2], i

            E.append(min_dist)

            add_in_V(vertexes, du[ind])

            du.pop(ind)

            remove_all_in_V_pass(vertexes, du)

        print(sum(E) / 100 - len(E))

    while True:

        n = eval(input())

        if n == 0:

            exit()

        m = eval(input())

        data = []

        for i in range(m):

            data.append(list(map(int, input().split(","))))

        solve(n, m, data)


problem_p00072()
