import math
import re


def fn_add_numbers(a: int | float, b: int | float) -> int | float:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Sum of both numbers.
    """
    return a + b


def fn_greet(name: str) -> str:
    """Generate a greeting message.

    Args:
        name: Person name.

    Returns:
        Greeting message.
    """
    return f"Hello, {name}"


def fn_reverse_string(s: str) -> str:
    """Reverse a string.

    Args:
        s: String to reverse.

    Returns:
        Reversed string.
    """
    return s[::-1]


def fn_get_square_root(a: int | float) -> int | float:
    """Calculate a square root.

    Args:
        a: Number whose square root should be calculated.

    Returns:
        Square root of the number.
    """
    return math.sqrt(a)


def fn_substitute_string_with_regex(
        source_string: str,
        regex: str,
        replacement: str,
) -> str:
    """Substitute regex matches in a string.

    Args:
        source_string: Original string where replacement happens.
        regex: Regex pattern to match.
        replacement: Text used to replace each match.

    Returns:
        String after regex substitution.
    """
    return re.sub(regex, replacement, source_string)


if __name__ == "__main__":
    print(fn_reverse_string("abcde"))
