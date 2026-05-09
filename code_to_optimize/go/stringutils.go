package sample

import "strings"

func ReverseString(s string) string {
	result := ""
	for i := len(s) - 1; i >= 0; i-- {
		result = result + string(s[i])
	}
	return result
}

func IsPalindrome(s string) bool {
	reversed := ReverseString(s)
	return s == reversed
}

func CountWords(s string) int {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return 0
	}
	return len(strings.Fields(trimmed))
}

func CapitalizeWords(s string) string {
	if s == "" {
		return s
	}

	words := strings.Split(s, " ")
	result := ""

	for i, word := range words {
		if len(word) > 0 {
			capitalized := strings.ToUpper(word[:1]) + strings.ToLower(word[1:])
			result = result + capitalized
		}
		if i < len(words)-1 {
			result = result + " "
		}
	}
	return result
}

func CountOccurrences(s, sub string) int {
	if sub == "" {
		return 0
	}

	count := 0
	index := 0
	for {
		pos := strings.Index(s[index:], sub)
		if pos == -1 {
			break
		}
		count++
		index = index + pos + 1
	}
	return count
}

func RemoveWhitespace(s string) string {
	result := ""
	for _, c := range s {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			result = result + string(c)
		}
	}
	return result
}

func FindAllIndices(s string, c byte) []int {
	var indices []int
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			indices = append(indices, i)
		}
	}
	return indices
}

func IsNumeric(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func Repeat(s string, n int) string {
	if n <= 0 {
		return ""
	}
	result := ""
	for i := 0; i < n; i++ {
		result = result + s
	}
	return result
}

func Truncate(s string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

func ToTitleCase(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + strings.ToLower(s[1:])
}
