# pylint: disable-all

from unittest import TestCase


class hroTestCase(TestCase):

    def assert_list_almost_equal(
            self, result, expected_result, places, msg=""
    ):
        for i, element in enumerate(expected_result):
            try:
                if result[i] is None:
                    self.assertIsNone(element)
                    continue
                if element is None:
                    self.assertIsNone(result[i])
                    continue

                self.assertAlmostEqual(result[i], element, places, msg)
            except AssertionError as e:
                raise AssertionError(
                    f"{result} != {expected_result},\n"
                    "first different entry is "
                    f"{result[i]} != {expected_result[i]} at index {i}.\n\n"
                    f"{msg}"
                ) from e

    def assert_time_list_equal(self, result, expected, msg=""):
        for i, (res, exp) in enumerate(zip(result, expected)):
            try:
                if result[i] is None:
                    self.assertIsNone(exp)
                    continue

                if exp is None:
                    self.assertIsNone(result[i])
                    continue

            except AssertionError as e:
                raise AssertionError(
                    f"{result} != {expected},\n"
                    "first different entry is "
                    f"{result[i]} != {expected[i]} at index {i}.\n\n"
                    f"{msg}"
                ) from e

            is_equal = (
                    (res.hour == exp.hour)
                    and (res.minute == exp.minute)
                    and (res.second == exp.second)
                    and (res.microsecond == exp.microsecond)
            )
            if not is_equal:
                raise AssertionError(
                    f"{result} != {expected},\n"
                    "first different entry is "
                    f"{result[i]} != {expected[i]} at index {i}.\n\n"
                    f"{msg}"
                )


def decorator_with_params(decorator):
    def wrapper(*args, **kwargs):
        def insert_arguments_in_decorator(f):
            return decorator(f, *args, **kwargs)

        return insert_arguments_in_decorator

    return wrapper


@decorator_with_params
def parameterized(test_method, testdata):

    def wrapper(self):
        failed_testcases = []
        for index, testcase in enumerate(testdata):
            try:
                test_method(self, *testcase)
            except Exception as e:
                failed_testcases.append((index, testcase, e))

        if len(failed_testcases) > 0:
            raise Exception(_exception_string(failed_testcases))

    def _exception_string_for_single_testcase(index, failed_testcase, exception):
        return f"testcase {index} - {failed_testcase}: {repr(exception)}"

    def _exception_string(failed_testcases):
        prefix = f"There has been a failing testcase: \n"\
            if len(failed_testcases) == 1 \
            else f"There have been {len(failed_testcases)} failing testcases: \n"
        failed_cases_string = "\n".join(
            [_exception_string_for_single_testcase(*case) for case in
             failed_testcases]
        )

        return prefix + failed_cases_string

    return wrapper
