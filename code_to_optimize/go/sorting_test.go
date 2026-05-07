package sample

import (
	"reflect"
	"testing"
)

func TestBubbleSort(t *testing.T) {
	tests := []struct {
		input    []int
		expected []int
	}{
		{[]int{5, 3, 1, 4, 2}, []int{1, 2, 3, 4, 5}},
		{[]int{3, 2, 1}, []int{1, 2, 3}},
		{[]int{1}, []int{1}},
		{[]int{}, []int{}},
		{[]int{1, 2, 3, 4, 5}, []int{1, 2, 3, 4, 5}},
	}

	for _, tc := range tests {
		result := BubbleSort(tc.input)
		if !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("BubbleSort(%v) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestBubbleSortWithDuplicates(t *testing.T) {
	result := BubbleSort([]int{3, 2, 4, 1, 3, 2})
	expected := []int{1, 2, 2, 3, 3, 4}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("got %v, want %v", result, expected)
	}
}

func TestBubbleSortWithNegatives(t *testing.T) {
	result := BubbleSort([]int{3, -2, 7, 0, -5})
	expected := []int{-5, -2, 0, 3, 7}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("got %v, want %v", result, expected)
	}
}

func TestBubbleSortDescending(t *testing.T) {
	tests := []struct {
		input    []int
		expected []int
	}{
		{[]int{1, 3, 5, 2, 4}, []int{5, 4, 3, 2, 1}},
		{[]int{1, 2, 3}, []int{3, 2, 1}},
		{[]int{}, []int{}},
	}

	for _, tc := range tests {
		result := BubbleSortDescending(tc.input)
		if !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("BubbleSortDescending(%v) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestInsertionSort(t *testing.T) {
	tests := []struct {
		input    []int
		expected []int
	}{
		{[]int{5, 3, 1, 4, 2}, []int{1, 2, 3, 4, 5}},
		{[]int{3, 2, 1}, []int{1, 2, 3}},
		{[]int{1}, []int{1}},
		{[]int{}, []int{}},
	}

	for _, tc := range tests {
		result := InsertionSort(tc.input)
		if !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("InsertionSort(%v) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestSelectionSort(t *testing.T) {
	tests := []struct {
		input    []int
		expected []int
	}{
		{[]int{5, 3, 1, 4, 2}, []int{1, 2, 3, 4, 5}},
		{[]int{3, 2, 1}, []int{1, 2, 3}},
		{[]int{1}, []int{1}},
	}

	for _, tc := range tests {
		result := SelectionSort(tc.input)
		if !reflect.DeepEqual(result, tc.expected) {
			t.Errorf("SelectionSort(%v) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestIsSorted(t *testing.T) {
	if !IsSorted([]int{1, 2, 3, 4, 5}) {
		t.Error("expected sorted")
	}
	if !IsSorted([]int{1}) {
		t.Error("expected sorted")
	}
	if !IsSorted([]int{}) {
		t.Error("expected sorted")
	}
	if IsSorted([]int{5, 3, 1}) {
		t.Error("expected not sorted")
	}
}

func TestBubbleSortDoesNotMutateInput(t *testing.T) {
	original := []int{5, 3, 1, 4, 2}
	saved := make([]int, len(original))
	copy(saved, original)
	BubbleSort(original)
	if !reflect.DeepEqual(original, saved) {
		t.Errorf("input was mutated: got %v, want %v", original, saved)
	}
}
