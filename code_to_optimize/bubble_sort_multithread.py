# from code_to_optimize.bubble_sort_codeflash_trace import sorter
from code_to_optimize.bubble_sort_codeflash_trace import sorter
import concurrent.futures


def multithreaded_sorter(unsorted_lists: list[list[int]]) -> list[list[int]]:
    # Create a list to store results in the correct order
    sorted_lists = [None] * len(unsorted_lists)

    # Use ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all sorting tasks and map them to their original indices
        future_to_index = {
            executor.submit(sorter, unsorted_list): i
            for i, unsorted_list in enumerate(unsorted_lists)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            sorted_lists[index] = future.result()

    return sorted_lists