def problem_p03080():
    def red_or_blue(N: int, s: str) -> bool:

        red = sum(c == "R" for c in s)

        return red > N - red

    if __name__ == "__main__":

        N = int(eval(input()))

        s = eval(input())

        yes = red_or_blue(N, s)

        print(("Yes" if yes else "No"))


problem_p03080()
