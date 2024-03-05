def solve(s):
    f_found = False
    for i in range(len(s) - 1, -1, -1):
        if s[i] == 'F':
            f_found = True
        if s[i] == 'C' and f_found:
            return "Yes"
    return "No"

def main():
    s = input()
    result = solve(s)
    print(result)

if __name__ == "__main__":
    main()
