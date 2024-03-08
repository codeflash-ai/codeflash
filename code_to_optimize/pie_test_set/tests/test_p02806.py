from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02806_0():
    input_content = "3\ndwango 2\nsixth 5\nprelims 25\ndwango"
    expected_output = "30"
    run_pie_test_case("../p02806.py", input_content, expected_output)


def test_problem_p02806_1():
    input_content = "15\nypnxn 279\nkgjgwx 464\nqquhuwq 327\nrxing 549\npmuduhznoaqu 832\ndagktgdarveusju 595\nwunfagppcoi 200\ndhavrncwfw 720\njpcmigg 658\nwrczqxycivdqn 639\nmcmkkbnjfeod 992\nhtqvkgkbhtytsz 130\ntwflegsjz 467\ndswxxrxuzzfhkp 989\nszfwtzfpnscgue 958\npmuduhznoaqu"
    expected_output = "6348"
    run_pie_test_case("../p02806.py", input_content, expected_output)


def test_problem_p02806_2():
    input_content = "3\ndwango 2\nsixth 5\nprelims 25\ndwango"
    expected_output = "30"
    run_pie_test_case("../p02806.py", input_content, expected_output)


def test_problem_p02806_3():
    input_content = "1\nabcde 1000\nabcde"
    expected_output = "0"
    run_pie_test_case("../p02806.py", input_content, expected_output)
