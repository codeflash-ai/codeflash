def problem_p03601(input_data):
    from fractions import gcd

    def solve():

        A, B, C, D, E, F = list(map(int, input_data.split()))

        ans = {"w": A * 100, "s": 0}

        def density(water, sugar):

            return 100 * sugar / (water + sugar)

        G = gcd(C, D)

        for a in range(31):

            for b in range(31):

                water = (a * A + b * B) * 100

                if water > F:

                    continue

                max_sugar = min((a * A + b * B) * E, F - water)

                sugar = int(G * (max_sugar // G))

                if sugar < min(C, D):
                    continue

                if water + sugar == 0:

                    continue

                if density(water, sugar) > density(ans["w"], ans["s"]):

                    ans["w"] = water

                    ans["s"] = sugar

                    # return (water, sugar)

        return (ans["w"] + ans["s"], ans["s"])

    if __name__ == "__main__":

        solve()
