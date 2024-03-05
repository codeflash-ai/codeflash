def main():
    s = [[0]*4 for _ in range(4)]
    for i in range(1, 4):
        s[i][1:4] = map(int, input().split())

    n = int(input())
    b = [[0]*4 for _ in range(4)]
    for _ in range(n):
        x = int(input())
        for i in range(1, 4):
            for j in range(1, 4):
                if s[i][j] == x and b[i][j] == 0:
                    b[i][j] = 1
                    break

    flag = 0
    for i in range(1, 4):
        if all(b[i][j] == 1 for j in range(1, 4)):
            flag = 1
            break
        if all(b[j][i] == 1 for j in range(1, 4)):
            flag = 1
            break

    if b[1][1] == 1 and b[2][2] == 1 and b[3][3] == 1:
        flag = 1
    if b[1][3] == 1 and b[2][2] == 1 and b[3][1] == 1:
        flag = 1

    if flag == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
