import sys

from codeflash.verification.codeflash_capture import codeflash_capture


class BubbleSorter:

    @codeflash_capture(function_name='BubbleSorter.__init__', tmp_dir_path='/var/folders/mg/k_c0twcj37q_gph3cfy3zlt80000gn/T/codeflash_ec8xrcji/test_return_values', tests_root='/Users/krrt7/Desktop/work/cf_org/codeflash/code_to_optimize/tests/pytest', is_fto=True)
    def __init__(self, x=0):
        self.x = x

    def sorter(self, arr):
        print('codeflash stdout : BubbleSorter.sorter() called')
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print('stderr test', file=sys.stderr)
        return arr

    @classmethod
    def sorter_classmethod(cls, arr):
        print('codeflash stdout : BubbleSorter.sorter_classmethod() called')
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print('stderr test classmethod', file=sys.stderr)
        return arr

    @staticmethod
    def sorter_staticmethod(arr):
        print('codeflash stdout : BubbleSorter.sorter_staticmethod() called')
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        print('stderr test staticmethod', file=sys.stderr)
        return arr
