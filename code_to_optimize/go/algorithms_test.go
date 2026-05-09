package sample

import (
	"reflect"
	"testing"
)

func TestTwoSum(t *testing.T) {
	got := TwoSum([]int{2, 7, 11, 15}, 9)
	if got != [2]int{0, 1} {
		t.Errorf("TwoSum([2,7,11,15], 9) = %v, want [0,1]", got)
	}

	got = TwoSum([]int{1, 2, 3}, 10)
	if got != [2]int{-1, -1} {
		t.Errorf("TwoSum no match = %v, want [-1,-1]", got)
	}
}

func TestFindDuplicates(t *testing.T) {
	got := FindDuplicates([]int{1, 2, 3, 2, 4, 3, 5})
	want := []int{2, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FindDuplicates = %v, want %v", got, want)
	}

	got = FindDuplicates([]int{1, 2, 3})
	if len(got) != 0 {
		t.Errorf("expected no duplicates, got %v", got)
	}
}

func TestUniqueElements(t *testing.T) {
	got := UniqueElements([]int{1, 2, 2, 3, 3, 3, 4})
	want := []int{1, 2, 3, 4}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("UniqueElements = %v, want %v", got, want)
	}
}

func TestMostFrequent(t *testing.T) {
	got := MostFrequent([]int{1, 2, 2, 3, 3, 3, 2, 2})
	if got != 2 {
		t.Errorf("MostFrequent = %d, want 2", got)
	}

	got = MostFrequent([]int{})
	if got != 0 {
		t.Errorf("MostFrequent empty = %d, want 0", got)
	}
}

func TestIntersection(t *testing.T) {
	got := Intersection([]int{1, 2, 3, 4}, []int{3, 4, 5, 6})
	want := []int{3, 4}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Intersection = %v, want %v", got, want)
	}

	got = Intersection([]int{1, 2}, []int{3, 4})
	if len(got) != 0 {
		t.Errorf("expected empty intersection, got %v", got)
	}
}

func TestMergeSortedSlices(t *testing.T) {
	got := MergeSortedSlices([]int{1, 3, 5}, []int{2, 4, 6})
	want := []int{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MergeSortedSlices = %v, want %v", got, want)
	}
}

func TestLongestCommonPrefix(t *testing.T) {
	got := LongestCommonPrefix([]string{"flower", "flow", "flight"})
	if got != "fl" {
		t.Errorf("LongestCommonPrefix = %q, want \"fl\"", got)
	}

	got = LongestCommonPrefix([]string{"dog", "racecar", "car"})
	if got != "" {
		t.Errorf("LongestCommonPrefix = %q, want \"\"", got)
	}

	got = LongestCommonPrefix([]string{})
	if got != "" {
		t.Errorf("LongestCommonPrefix empty = %q, want \"\"", got)
	}
}

func TestMaxSubarraySum(t *testing.T) {
	got := MaxSubarraySum([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4})
	if got != 6 {
		t.Errorf("MaxSubarraySum = %d, want 6", got)
	}

	got = MaxSubarraySum([]int{-1, -2, -3})
	if got != -1 {
		t.Errorf("MaxSubarraySum all negative = %d, want -1", got)
	}

	got = MaxSubarraySum([]int{})
	if got != 0 {
		t.Errorf("MaxSubarraySum empty = %d, want 0", got)
	}
}

func TestIsPrime(t *testing.T) {
	primes := []int{2, 3, 5, 7, 11, 13, 17, 19, 23}
	for _, p := range primes {
		if !IsPrime(p) {
			t.Errorf("IsPrime(%d) = false, want true", p)
		}
	}

	nonPrimes := []int{0, 1, 4, 6, 8, 9, 10, 15}
	for _, n := range nonPrimes {
		if IsPrime(n) {
			t.Errorf("IsPrime(%d) = true, want false", n)
		}
	}
}

func TestPrimesUpTo(t *testing.T) {
	got := PrimesUpTo(20)
	want := []int{2, 3, 5, 7, 11, 13, 17, 19}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PrimesUpTo(20) = %v, want %v", got, want)
	}
}

func TestGCD(t *testing.T) {
	tests := []struct {
		a, b, want int
	}{
		{12, 8, 4},
		{7, 13, 1},
		{0, 5, 5},
		{-12, 8, 4},
	}

	for _, tc := range tests {
		got := GCD(tc.a, tc.b)
		if got != tc.want {
			t.Errorf("GCD(%d, %d) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestLCM(t *testing.T) {
	tests := []struct {
		a, b, want int
	}{
		{4, 6, 12},
		{7, 13, 91},
		{0, 5, 0},
	}

	for _, tc := range tests {
		got := LCM(tc.a, tc.b)
		if got != tc.want {
			t.Errorf("LCM(%d, %d) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}
