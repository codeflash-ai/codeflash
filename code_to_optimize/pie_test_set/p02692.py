def problem_p02692():
    def solve(writer):

        N, A, B, C = list(map(int, input().split()))

        variables = {"A": A, "B": B, "C": C}

        queries = (list(eval(input())) for _ in range(N))

        buffer = list()

        for i in range(N):

            if not buffer:

                t1, t2 = next(queries)

            else:

                b = buffer.pop()

                if b == "pass":

                    continue

                else:

                    t1, t2 = list(b)

            v1 = variables[t1]

            v2 = variables[t2]

            if v1 == v2 == 0:

                return False

            elif v1 == v2 == 1:

                if i == N - 1:

                    writer.append(t1)

                    return True

                nt1, nt2 = next(queries)

                if len({t1, t2, nt1, nt2}) == 2:

                    buffer.append("pass")

                    writer.append(t1)

                    writer.append(t2)

                    continue

                if t1 in {nt1, nt2}:

                    variables[t1] += 1

                    variables[t2] -= 1

                    writer.append(t1)

                else:

                    variables[t1] -= 1

                    variables[t2] += 1

                    writer.append(t2)

                buffer.append("".join([nt1, nt2]))

            elif v1 > v2:

                variables[t1] -= 1

                variables[t2] += 1

                writer.append(t2)

            elif v2 > v1:

                variables[t1] += 1

                variables[t2] -= 1

                writer.append(t1)

            else:

                variables[t1] += 1

                variables[t2] -= 1

                writer.append(t1)

        return True

    def main():

        writer = list()

        ok = solve(writer)

        if ok:

            print("Yes")

            for w in writer:

                print(w)

        else:

            print("No")

    if __name__ == "__main__":

        main()


problem_p02692()
