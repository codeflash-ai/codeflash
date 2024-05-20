def concatenate_strings(n):
    result = ""
    for i in range(n):
        result += str(i) + ", "
    return result


def process_tasks(task_data):
    """
    Processes a list of tasks categorized by type and logs the processing.

    :param task_data: Dictionary with task categories as keys and number of tasks as values.
    :return: A formatted log string and summary of the tasks.
    """
    log = ""
    total_tasks = 0
    category_summary = []

    # Process each category of tasks
    for category, num_tasks in task_data.items():
        # Process each task using the concatenation function
        category_log = concatenate_strings(num_tasks)

        # Clean up the log string for the category
        clean_log = category_log[:-2] + "."
        log += f"Category '{category}': {clean_log}\n"

        # Update total tasks and prepare summary data
        total_tasks += num_tasks
        category_summary.append((category, num_tasks))

    # Generate summary statistics
    summary = f"Total tasks processed: {total_tasks}\n"
    for category, count in category_summary:
        summary += f"{category}: {count} tasks, representing {count / total_tasks:.2%} of total.\n"

    # Return both detailed log and summary
    return log, summary
