from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00141_0():
    input_content = "2\n5\n6"
    expected_output = (
        "#####\n#   #\n# # #\n# # #\n# ###\n\n######\n#    #\n# ## #\n# #  #\n# #  #\n# ####"
    )
    run_pie_test_case("../p00141.py", input_content, expected_output)


def test_problem_p00141_1():
    input_content = "2\n5\n6"
    expected_output = (
        "#####\n#   #\n# # #\n# # #\n# ###\n\n######\n#    #\n# ## #\n# #  #\n# #  #\n# ####"
    )
    run_pie_test_case("../p00141.py", input_content, expected_output)
