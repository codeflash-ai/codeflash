def problem_p03631(input_data):
    def is_palindrome(s: str) -> bool:

        return s[: len(s) // 2] == s[-1 * (len(s) // 2) :][::-1]

    N = eval(input_data)

    return "Yes" if is_palindrome(N) else "No"
