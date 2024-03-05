def main():
    n = int(input())
    a = [0] * 100005
    for i in range(n):
        a[i] = int(input())
    a = sorted(a[:n])
    x = 0
    j = 0
    while j < n:
        if j + 1 < n and a[j] == a[j + 1]:
            j += 2
        else:
            j += 1
            x += 1
    print(x)

if __name__ == "__main__":
    main()
