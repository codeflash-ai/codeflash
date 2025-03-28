# import unittest
#
# from parameterized import parameterized
#
# from code_to_optimize.bubble_sort import sorter
#
#
# class TestPigLatin(unittest.TestCase):
#     @parameterized.expand(
#         [
#             ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
#             ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
#             (list(reversed(range(50))), list(range(50))),
#         ]
#     )
#     def test_sort(self, input, expected_output):
#         output = sorter(input)
#         self.assertEqual(output, expected_output)
