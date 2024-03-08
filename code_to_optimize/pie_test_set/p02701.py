def problem_p02701():
    N = int(eval(input()))

    S = [eval(input()) for i in range(N)]

    from collections import Counter

    Sc = Counter(S).most_common()

    print((len(Sc)))


problem_p02701()
