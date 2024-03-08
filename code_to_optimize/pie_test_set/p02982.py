def problem_p02982():
    from scipy.spatial import distance

    def solve(string):

        n, d, *x = list(map(int, string.split()))

        x = [tuple(c) for c in zip(*[iter(x)] * d)]

        dist = distance.cdist(x, x)

        return str(((dist == dist.astype("int32")).sum() - n) // 2)

    if __name__ == "__main__":

        n, m = list(map(int, input().split()))

        print((solve("{} {}\n".format(n, m) + "\n".join([eval(input()) for _ in range(n)]))))


problem_p02982()
