package sample

import (
	"reflect"
	"testing"
)

func TestReverseString(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello", "olleh"},
		{"a", "a"},
		{"", ""},
		{"abcd", "dcba"},
	}

	for _, tc := range tests {
		got := ReverseString(tc.input)
		if got != tc.want {
			t.Errorf("ReverseString(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestIsPalindrome(t *testing.T) {
	palindromes := []string{"racecar", "madam", "a", "", "abba"}
	for _, s := range palindromes {
		if !IsPalindrome(s) {
			t.Errorf("IsPalindrome(%q) = false, want true", s)
		}
	}

	nonPalindromes := []string{"hello", "ab"}
	for _, s := range nonPalindromes {
		if IsPalindrome(s) {
			t.Errorf("IsPalindrome(%q) = true, want false", s)
		}
	}
}

func TestCountWords(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"hello world test", 3},
		{"hello", 1},
		{"", 0},
		{"   ", 0},
		{"  multiple   spaces   between   words  ", 4},
	}

	for _, tc := range tests {
		got := CountWords(tc.input)
		if got != tc.want {
			t.Errorf("CountWords(%q) = %d, want %d", tc.input, got, tc.want)
		}
	}
}

func TestCapitalizeWords(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello world", "Hello World"},
		{"HELLO", "Hello"},
		{"", ""},
		{"one two three", "One Two Three"},
	}

	for _, tc := range tests {
		got := CapitalizeWords(tc.input)
		if got != tc.want {
			t.Errorf("CapitalizeWords(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestCountOccurrences(t *testing.T) {
	tests := []struct {
		s, sub string
		want   int
	}{
		{"hello hello", "hello", 2},
		{"aaa", "a", 3},
		{"aaa", "aa", 2},
		{"hello", "world", 0},
		{"hello", "", 0},
	}

	for _, tc := range tests {
		got := CountOccurrences(tc.s, tc.sub)
		if got != tc.want {
			t.Errorf("CountOccurrences(%q, %q) = %d, want %d", tc.s, tc.sub, got, tc.want)
		}
	}
}

func TestRemoveWhitespace(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello world", "helloworld"},
		{"  a b c  ", "abc"},
		{"test", "test"},
		{"   ", ""},
		{"", ""},
	}

	for _, tc := range tests {
		got := RemoveWhitespace(tc.input)
		if got != tc.want {
			t.Errorf("RemoveWhitespace(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestFindAllIndices(t *testing.T) {
	got := FindAllIndices("hello", 'l')
	want := []int{2, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FindAllIndices(\"hello\", 'l') = %v, want %v", got, want)
	}

	got = FindAllIndices("aaa", 'a')
	if len(got) != 3 {
		t.Errorf("expected 3 indices, got %d", len(got))
	}

	got = FindAllIndices("hello", 'z')
	if len(got) != 0 {
		t.Errorf("expected 0 indices, got %d", len(got))
	}

	got = FindAllIndices("", 'a')
	if len(got) != 0 {
		t.Errorf("expected 0 indices, got %d", len(got))
	}
}

func TestIsNumeric(t *testing.T) {
	numerics := []string{"12345", "0", "007"}
	for _, s := range numerics {
		if !IsNumeric(s) {
			t.Errorf("IsNumeric(%q) = false, want true", s)
		}
	}

	nonNumerics := []string{"12.34", "-123", "abc", "12a34", ""}
	for _, s := range nonNumerics {
		if IsNumeric(s) {
			t.Errorf("IsNumeric(%q) = true, want false", s)
		}
	}
}

func TestRepeat(t *testing.T) {
	tests := []struct {
		s    string
		n    int
		want string
	}{
		{"abc", 3, "abcabcabc"},
		{"a", 3, "aaa"},
		{"abc", 0, ""},
		{"abc", -1, ""},
	}

	for _, tc := range tests {
		got := Repeat(tc.s, tc.n)
		if got != tc.want {
			t.Errorf("Repeat(%q, %d) = %q, want %q", tc.s, tc.n, got, tc.want)
		}
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		s       string
		maxLen  int
		want    string
	}{
		{"hello", 10, "hello"},
		{"hello world", 6, "hel..."},
		{"hello world", 8, "hello..."},
		{"hello", 0, ""},
		{"hello", 3, "hel"},
	}

	for _, tc := range tests {
		got := Truncate(tc.s, tc.maxLen)
		if got != tc.want {
			t.Errorf("Truncate(%q, %d) = %q, want %q", tc.s, tc.maxLen, got, tc.want)
		}
	}
}

func TestToTitleCase(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"hello", "Hello"},
		{"HELLO", "Hello"},
		{"hELLO", "Hello"},
		{"a", "A"},
		{"", ""},
	}

	for _, tc := range tests {
		got := ToTitleCase(tc.input)
		if got != tc.want {
			t.Errorf("ToTitleCase(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
