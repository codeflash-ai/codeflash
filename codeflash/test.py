class Solution:
    _vowels = set("aeiouAEIOU")  # noqa: RUF012

    def reverseVowels(self, s: str) -> str:
        left, right = 0, len(s) - 1
        chars = list(s)
        vowels = self._vowels  # Local variable for faster lookup

        while left < right:
            # Advance left pointer to next vowel
            while left < right and chars[left] not in vowels:
                left += 1
            # Advance right pointer to previous vowel
            while left < right and chars[right] not in vowels:
                right -= 1
            if left < right:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1
                right -= 1

        return "".join(chars)
