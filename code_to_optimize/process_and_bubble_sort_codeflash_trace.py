from code_to_optimize.bubble_sort import sorter
from codeflash.benchmarking.codeflash_trace import codeflash_trace


def calculate_pairwise_products(arr):
    """Calculate the average of all pairwise products in the array."""
    if len(arr) < 2:
        return 0
    total = sum(arr)
    total_sq = sum(x * x for x in arr)
    sum_of_products = total * total - total_sq
    count = len(arr) * (len(arr) - 1)
    return sum_of_products / count


@codeflash_trace
def compute_and_sort(arr):
    # Compute pairwise sums average
    pairwise_average = calculate_pairwise_products(arr)

    # Call sorter function
    sorter(arr.copy())

    return pairwise_average
