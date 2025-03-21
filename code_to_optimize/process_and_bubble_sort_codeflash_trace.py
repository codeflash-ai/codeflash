from code_to_optimize.bubble_sort import sorter
from codeflash.benchmarking.codeflash_trace import codeflash_trace

def calculate_pairwise_products(arr):
    """
    Calculate the average of all pairwise products in the array.
    """
    sum_of_products = 0
    count = 0

    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j:
                sum_of_products += arr[i] * arr[j]
                count += 1

    # The average of all pairwise products
    return sum_of_products / count if count > 0 else 0

@codeflash_trace
def compute_and_sort(arr):
    # Compute pairwise sums average
    pairwise_average = calculate_pairwise_products(arr)

    # Call sorter function
    sorter(arr.copy())

    return pairwise_average
