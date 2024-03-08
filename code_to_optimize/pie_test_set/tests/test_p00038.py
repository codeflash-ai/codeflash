from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00038_0():
    input_content = "1,2,3,4,1\n2,3,2,3,12\n12,13,11,12,12\n7,6,7,6,7\n3,3,2,3,3\n6,7,8,9,10\n11,12,10,1,13\n11,12,13,1,2"
    expected_output = (
        "one pair\ntwo pair\nthree card\nfull house\nfour card\nstraight\nstraight\nnull"
    )
    run_pie_test_case("../p00038.py", input_content, expected_output)


def test_problem_p00038_1():
    input_content = "1,2,3,4,1\n2,3,2,3,12\n12,13,11,12,12\n7,6,7,6,7\n3,3,2,3,3\n6,7,8,9,10\n11,12,10,1,13\n11,12,13,1,2"
    expected_output = (
        "one pair\ntwo pair\nthree card\nfull house\nfour card\nstraight\nstraight\nnull"
    )
    run_pie_test_case("../p00038.py", input_content, expected_output)
