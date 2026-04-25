package sample

func BubbleSort(arr []int) []int {
	if len(arr) == 0 {
		return arr
	}

	result := make([]int, len(arr))
	copy(result, arr)
	n := len(result)

	for i := 0; i < n; i++ {
		for j := 0; j < n-1; j++ {
			if result[j] > result[j+1] {
				temp := result[j]
				result[j] = result[j+1]
				result[j+1] = temp
			}
		}
	}
	return result
}

func BubbleSortDescending(arr []int) []int {
	if len(arr) == 0 {
		return arr
	}

	result := make([]int, len(arr))
	copy(result, arr)
	n := len(result)

	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if result[j] < result[j+1] {
				temp := result[j]
				result[j] = result[j+1]
				result[j+1] = temp
			}
		}
	}
	return result
}

func InsertionSort(arr []int) []int {
	if len(arr) == 0 {
		return arr
	}

	result := make([]int, len(arr))
	copy(result, arr)
	n := len(result)

	for i := 1; i < n; i++ {
		key := result[i]
		j := i - 1
		for j >= 0 && result[j] > key {
			result[j+1] = result[j]
			j--
		}
		result[j+1] = key
	}
	return result
}

func SelectionSort(arr []int) []int {
	if len(arr) == 0 {
		return arr
	}

	result := make([]int, len(arr))
	copy(result, arr)
	n := len(result)

	for i := 0; i < n-1; i++ {
		minIdx := i
		for j := i + 1; j < n; j++ {
			if result[j] < result[minIdx] {
				minIdx = j
			}
		}
		result[minIdx], result[i] = result[i], result[minIdx]
	}
	return result
}

func IsSorted(arr []int) bool {
	for i := 0; i < len(arr)-1; i++ {
		if arr[i] > arr[i+1] {
			return false
		}
	}
	return true
}
