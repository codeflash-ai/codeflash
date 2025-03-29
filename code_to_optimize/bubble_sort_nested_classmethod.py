from code_to_optimize.bubble_sort_in_nested_class import WrapperClass
from line_profiler import profile as codeflash_line_profile


@codeflash_line_profile
def sort_classmethod(x):
    y = WrapperClass.BubbleSortClass()
    return y.sorter(x)
