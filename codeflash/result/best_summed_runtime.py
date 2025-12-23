def calculate_best_summed_runtime(grouped_runtime_info: dict[any, list[int]]) -> int:
    return sum([min(usable_runtime_data) for _, usable_runtime_data in grouped_runtime_info.items()])
