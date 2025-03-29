from code_to_optimize.bubble_sort_in_nested_class import WrapperClass


def sort_classmethod(x):
    y = WrapperClass.BubbleSortClass()
    return y.sorter(x)
