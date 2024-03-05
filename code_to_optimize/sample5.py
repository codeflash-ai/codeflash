MOD = 100000000000000003
MA = (1 << 19) - 1
NUM = [0] * 524300
KAZU = [0] * 524300

def query(n):
    n2 = (n & MA) + 1
    while NUM[n2]:
        if NUM[n2] == n + 1:
            return KAZU[n2]
        n2 = n2 * 61 % 524287
    return 0

def add(n, b):
    n2 = (n & MA) + 1
    while NUM[n2]:
        if NUM[n2] == n + 1:
            KAZU[n2] |= b
            return
        n2 = n2 * 61 % 524287
    NUM[n2] = n + 1
    KAZU[n2] = b

def main():
    input_data = input().read()
    ci = iter(input_data)
    N = int(next(ci))
    ST = set()
    s = [''] * (N + 1)
    for i in range(1, N + 1):
        s[i] = next(ci)
        n = len(s[i])
        if n <= 100000:
            ne[i] = he[n]
            he[n] = i
        else:
            ST.add((n, s[i]))

    answer = 0
    for M in range(1, 100001):
        p = he[M]
        while p:
            S = s[p]
            k = [0] * 26
            are = 0
            for i in range(M - 1, 0, -1):
                tmp = query(are)
                if tmp:
                    for j in range(26):
                        if (tmp >> j) & 1:
                            k[j] += 1
                answer += k[ord(S[i]) - ord('a')]
                k[ord(S[i]) - ord('a')] = 0
                are = (are * 30 + ord(S[i]) - ord('a') + 1) % MOD
            answer += k[ord(S[0]) - ord('a')]
            k[ord(S[0]) - ord('a')] = 0
            add(are, 1 << (ord(S[0]) - ord('a')))
            p = ne[p]

    for n, S in ST:
        k = [0] * 26
        are = 0
        for i in range(n - 1, 0, -1):
            tmp = query(are)
            if tmp:
                for j in range(26):
                    if (tmp >> j) & 1:
                        k[j] += 1
            answer += k[ord(S[i]) - ord('a')]
            k[ord(S[i]) - ord('a')] = 0
            are = (are * 30 + ord(S[i]) - ord('a') + 1) % MOD
        answer += k[ord(S[0]) - ord('a')]
        k[ord(S[0]) - ord('a')] = 0
        add(are, 1 << (ord(S[0]) - ord('a')))

    print(answer)

if __name__ == "__main__":
    main()
