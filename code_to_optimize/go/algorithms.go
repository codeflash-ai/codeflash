package sample

import "strings"

func TwoSum(nums []int, target int) [2]int {
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			if nums[i]+nums[j] == target {
				return [2]int{i, j}
			}
		}
	}
	return [2]int{-1, -1}
}

func FindDuplicates(nums []int) []int {
	var result []int
	for i := 0; i < len(nums); i++ {
		found := false
		for j := 0; j < i; j++ {
			if nums[i] == nums[j] {
				found = true
				break
			}
		}
		if found {
			alreadyAdded := false
			for _, r := range result {
				if r == nums[i] {
					alreadyAdded = true
					break
				}
			}
			if !alreadyAdded {
				result = append(result, nums[i])
			}
		}
	}
	return result
}

func UniqueElements(nums []int) []int {
	var result []int
	for _, num := range nums {
		found := false
		for _, r := range result {
			if r == num {
				found = true
				break
			}
		}
		if !found {
			result = append(result, num)
		}
	}
	return result
}

func MostFrequent(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	maxCount := 0
	maxNum := nums[0]

	for _, num := range nums {
		count := 0
		for _, other := range nums {
			if other == num {
				count++
			}
		}
		if count > maxCount {
			maxCount = count
			maxNum = num
		}
	}
	return maxNum
}

func Intersection(a, b []int) []int {
	var result []int
	for _, x := range a {
		for _, y := range b {
			if x == y {
				already := false
				for _, r := range result {
					if r == x {
						already = true
						break
					}
				}
				if !already {
					result = append(result, x)
				}
			}
		}
	}
	return result
}

func MergeSortedSlices(a, b []int) []int {
	var result []int
	result = append(result, a...)
	result = append(result, b...)

	for i := 0; i < len(result); i++ {
		for j := i + 1; j < len(result); j++ {
			if result[j] < result[i] {
				result[i], result[j] = result[j], result[i]
			}
		}
	}
	return result
}

func LongestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}

	prefix := strs[0]
	for _, s := range strs[1:] {
		for !strings.HasPrefix(s, prefix) {
			prefix = prefix[:len(prefix)-1]
			if prefix == "" {
				return ""
			}
		}
	}
	return prefix
}

func MaxSubarraySum(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	maxSum := nums[0]
	for i := 0; i < len(nums); i++ {
		for j := i; j < len(nums); j++ {
			sum := 0
			for k := i; k <= j; k++ {
				sum += nums[k]
			}
			if sum > maxSum {
				maxSum = sum
			}
		}
	}
	return maxSum
}

func IsPrime(n int) bool {
	if n < 2 {
		return false
	}
	for i := 2; i < n; i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func PrimesUpTo(limit int) []int {
	var primes []int
	for i := 2; i <= limit; i++ {
		if IsPrime(i) {
			primes = append(primes, i)
		}
	}
	return primes
}

func GCD(a, b int) int {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func LCM(a, b int) int {
	if a == 0 || b == 0 {
		return 0
	}
	return a / GCD(a, b) * b
}
