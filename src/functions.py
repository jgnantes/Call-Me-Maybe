import math
import re


def fn_add_numbers(a: int | float, b: int | float) -> int | float:
    """ """
    return a + b


def fn_greet(name: str) -> str:
    """ """
    return f"Hello, {name}"


def fn_reverse_string(s: str) -> str:
    """ """
    return s[::-1]


def fn_get_square_root(a: int | float) -> int | float:
    """ """
    return math.sqrt(a)


def fn_substitute_string_with_regex(
        source_string: str,
        regex: str,
        replacement: str,
) -> str:
    """ """
    return re.sub(regex, replacement, source_string)


if __name__ == "__main__":
    print(fn_reverse_string("abcde"))