from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, NewType, Optional, Protocol, TypeVar

try:
    from typing import _TypingBase  # type: ignore[attr-defined]
except ImportError:
    from typing import _Final as _TypingBase  # type: ignore[attr-defined]
typing_base = _TypingBase

_T = TypeVar("_T")


class Comparable(Protocol):
    def __lt__(self: _T, __other: _T) -> bool: ...


ComparableT = TypeVar("ComparableT", bound=Comparable)


def sorter(arr: list[ComparableT]) -> list[ComparableT]:
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    return arr


def sorter2(arr: list[ComparableT]) -> list[ComparableT]:
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def sorter3(arr: list[ComparableT]) -> list[ComparableT]:
    arr.sort()
    return arr


def is_valid_field_name(name: str) -> bool:
    return not name.startswith("_")


def is_valid_field_name2(name: str) -> bool:
    return not (name and name[0] == "_")


def is_self_type(tp: Any) -> bool:
    """Check if a given class is a Self type (from `typing` or `typing_extensions`)"""
    return isinstance(tp, typing_base) and getattr(tp, "_name", None) == "Self"


def is_self_type2(tp: Any) -> bool:
    """Check if a given class is a Self type (from `typing` or `typing_extensions`)"""
    if not isinstance(tp, _TypingBase):
        return False
    return tp._name == "Self" if hasattr(tp, "_name") else False


test_new_type = NewType("test_new_type", str)


def is_new_type(type_: type[Any]) -> bool:
    """Check whether type_ was created using typing.NewType.
    Can't use isinstance because it fails <3.10.
    """
    return isinstance(type_, test_new_type.__class__) and hasattr(type_, "__supertype__")  # type: ignore[arg-type]


def is_new_type2(type_: type[Any]) -> bool:
    """Check whether type_ was created using typing.NewType.
    Can't use isinstance because it fails <3.10.
    """
    return type(type_) is type(test_new_type) and hasattr(type_, "__supertype__")


def _to_str(
    size: int,
    suffixes: Iterable[str],
    base: int,
    *,
    precision: Optional[int] = 1,
    separator: Optional[str] = " ",
) -> str:
    if size == 1:
        return "1 byte"
    if size < base:
        return f"{size:,} bytes"

    for i, suffix in enumerate(suffixes, 2):  # noqa: B007
        unit = base**i
        if size < unit:
            break
    return "{:,.{precision}f}{separator}{}".format(
        (base * size / unit),
        suffix,
        precision=precision,
        separator=separator,
    )


# Given: (size=-1, suffixes=(), base=-1, precision=0, separator=None),
#    code_to_optimize.bubble_sort_typed._to_str : raises UnboundLocalError("cannot access local variable 'unit' where it is not associated with a value")
#   code_to_optimize.bubble_sort_typed._to_str2 : raises IndexError()


def _to_str2(
    size: int,
    suffixes: Iterable[str],
    base: int,
    *,
    precision: Optional[int] = 1,
    separator: Optional[str] = " ",
) -> str:
    if size == 1:
        return "1 byte"
    if size < base:
        return f"{size:,} bytes"

    unit = base
    for suffix in suffixes:
        unit *= base
        if size < unit:
            return f"{size / (unit / base):,.{precision}f}{separator}{suffix}"

    # Extra condition if size exceeds the largest unit
    return f"{size / (unit / base):,.{precision}f}{separator}{suffixes[-1]}"


def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0]["tags"]
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article["tags"]]
    return set(common_tags)


# crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags2
# Given: (articles=[{'tags': ['', '']}, {'tags': ['', '']}, {'tags': []}, {}]),
#   code_to_optimize.bubble_sort_typed.find_common_tags : returns set()
#  code_to_optimize.bubble_sort_typed.find_common_tags2 : raises KeyError()


def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags.intersection_update(article["tags"])
    return common_tags


# Given: (articles=[{'\x00\x00\x00\x00': [], 'tags': ['']}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {'tags': ['']}, {}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {}]),
#      code_to_optimize.bubble_sort_typed.find_common_tags : raises KeyError()
#   code_to_optimize.bubble_sort_typed.find_common_tags2_1 : returns set()


def find_common_tags2_1(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0].get("tags", []))
    for article in articles[1:]:
        common_tags.intersection_update(article.get("tags", []))
    return common_tags


#  % crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags2_2
# Given: (articles=[{'\x00\x00\x00\x00': [''], 'tags': ['']}, {'\x00\x00\x00\x00': [''], 'tags': ['']}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {'\x00\x00\x00\x00': [], '': []}, {'\x00\x00\x00\x00': [], 'tags': ['']}]),
#      code_to_optimize.bubble_sort_typed.find_common_tags : raises KeyError()
#   code_to_optimize.bubble_sort_typed.find_common_tags2_2 : returns set()
# (codeflash312) renaud@Renauds-Laptop codeflash %


def find_common_tags2_2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        if not common_tags:
            break
        common_tags.intersection_update(article["tags"])
    return common_tags


# % crosshair diffbehavior --max_uninteresting_iterations 128 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags2_3
# Given: (articles=[{'tags': ['', '']}, {'tags': ['', '']}, {'tags': []}, {}]),
#      code_to_optimize.bubble_sort_typed.find_common_tags : returns set()
#   code_to_optimize.bubble_sort_typed.find_common_tags2_3 : raises KeyError()
# Given: (articles=[{'\x00\x00\x00\x00': [], 'tags': []}, {'\x00\x00\x00\x00': [], 'tags': []}, {'\x00\x00\x00\x00': [], 'tags': []}, {'\x00\x00\x00\x00': []}, {}, {}]),
#      code_to_optimize.bubble_sort_typed.find_common_tags : returns set()
#   code_to_optimize.bubble_sort_typed.find_common_tags2_3 : raises KeyError()


def find_common_tags2_3(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        article_tags = article["tags"]  # Access 'tags' key to match KeyError behavior
        if not common_tags:
            continue  # Skip intersection but maintain KeyError on missing 'tags'
        common_tags.intersection_update(article_tags)
    return common_tags


def find_common_tags2_4(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        if common_tags:
            article_tags = article["tags"]  # Access 'tags' only if common_tags is not empty
            common_tags.intersection_update(article_tags)
        else:
            # Do not access article["tags"]; no KeyError is raised
            pass
    return common_tags


def find_common_tags2_5(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    # Initialize with the first article's tags, defaulting to an empty list if "tags" is missing
    common_tags = set(articles[0].get("tags", []))

    for article in articles[1:]:
        # Use .get("tags", []) to safely access tags, defaulting to an empty list if missing
        common_tags.intersection_update(article.get("tags", []))

        # Early exit if there are no common tags left
        if not common_tags:
            break

    return common_tags


def find_common_tags2_6(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    # Initialize with the first article's tags
    common_tags = set(articles[0]["tags"])  # Raises KeyError if "tags" is missing

    for article in articles[1:]:
        # Directly access "tags", maintaining behavior
        common_tags.intersection_update(article["tags"])

        # Early exit if no common tags remain
        if not common_tags:
            break

    return common_tags


def find_common_tags2_7(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    # Initialize with the first article's tags (raises KeyError if "tags" is missing)
    common_tags = set(articles[0]["tags"])

    for article in articles[1:]:
        if not common_tags:
            # If no common tags remain, no need to process further
            break

        # Access "tags" directly, maintaining original behavior (raises KeyError if missing)
        common_tags.intersection_update(article["tags"])

    return common_tags


def find_common_tags2_8(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    # Initialize with the first article's tags (raises KeyError if "tags" is missing)
    try:
        common_tags = set(articles[0]["tags"])
    except KeyError:
        raise KeyError("The first article is missing the 'tags' key.")

    for index, article in enumerate(articles[1:], start=2):
        try:
            tags = article["tags"]
        except KeyError:
            raise KeyError(f"Article at position {index} is missing the 'tags' key.")

        # Perform intersection with the current article's tags
        common_tags.intersection_update(tags)

    return common_tags


def find_common_tags2_9(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    # Initialize with the first article's tags (raises KeyError if "tags" is missing)
    common_tags = set(articles[0]["tags"])

    for article in articles[1:]:
        if not common_tags:
            # If no common tags remain, no need to process further
            break
        # Directly access "tags", allowing KeyError to propagate naturally
        common_tags.intersection_update(article["tags"])

    return common_tags


# crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags3
# Given: (articles=[{'tags': ['', '', '', '']}, {'tags': ['', '', '', '']}, {'tags': ['', '', '']}, {'tags': ['', '', '', '']}, {'tags': ['', '', '']}, {}]),
#   code_to_optimize.bubble_sort_typed.find_common_tags : raises KeyError()
#  code_to_optimize.bubble_sort_typed.find_common_tags3 : returns set()
# Given: (articles=[{'\x00\x00\x00\x00': ['', ''], 'tags': [], '': []}, {}, {'\x00\x00\x00\x00': ['', ''], '': []}, {'': []}, {'\x00\x00\x00\x00': ['', ''], 'tags': [], '': []}]),
#   code_to_optimize.bubble_sort_typed.find_common_tags : returns set()
#  code_to_optimize.bubble_sort_typed.find_common_tags3 : raises KeyError()


def find_common_tags3(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags.intersection_update(article["tags"])
        if not common_tags:
            break
    return common_tags


# % crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags4
# Given: (articles=[{'\x00\x00\x00\x00': ['', ''], 'tags': [], '': []}, {}, {'\x00\x00\x00\x00': ['', ''], '': []}, {'': []}, {'\x00\x00\x00\x00': ['', ''], 'tags': [], '': []}]),
#   code_to_optimize.bubble_sort_typed.find_common_tags : returns set()
#  code_to_optimize.bubble_sort_typed.find_common_tags4 : raises KeyError()


def find_common_tags4(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags &= set(article["tags"])
        if not common_tags:  # Early exit if no common tags.
            break
    return common_tags


def with_pattern(pattern: str, regex_group_count: int | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        func.pattern = pattern
        func.regex_group_count = regex_group_count
        return func

    return decorator


def with_pattern2(pattern: str, regex_group_count: int | None = None) -> Callable:
    return (
        lambda func: setattr(func, "pattern", pattern)
        or setattr(func, "regex_group_count", regex_group_count)
        or func
    )


"""
We have this original code:
```
def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0]["tags"]
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article["tags"]]
    return set(common_tags)
```

We generated optimized code:
```
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags.intersection_update(article["tags"])
    return common_tags
```

When analyzed the optimized code for correctness with Concolic Testing (using CrossHair DiffBehavior), we found the following counter-example:
```
crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags2

Given: (articles=[{'tags': ['', '']}, {'tags': ['', '']}, {'tags': []}, {}]), code_to_optimize.bubble_sort_typed.find_common_tags : returns set() code_to_optimize.bubble_sort_typed.find_common_tags2 : raises KeyError()
```

Fix find_common_tags2 for correctness relative to find_common_tags, making sure it is still optimized for speed.
"""

"""
To fix the `KeyError` in `find_common_tags2` and ensure it behaves like the original `find_common_tags`, we'll modify the code to handle cases where the `'tags'` key might be missing in some articles. In the original code, it seems that articles might not always have the `'tags'` key, and in such cases, the code should default to an empty list.

Here's the corrected and optimized version:

```python
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()
    
    common_tags = set(articles[0].get("tags", []))
    for article in articles[1:]:
        common_tags.intersection_update(article.get("tags", []))
    return common_tags
```

**Explanation:**

- **Use `get` with a default value:**
  - We replaced `article["tags"]` with `article.get("tags", [])`. This ensures that if the `'tags'` key is missing, it defaults to an empty list instead of raising a `KeyError`.
- **Initialize `common_tags` properly:**
  - We initialize `common_tags` with `set(articles[0].get("tags", []))` to handle cases where the first article might not have `'tags'`.
- **Maintain Performance Optimization:**
  - By using `set` operations, we keep the optimized performance of the function while ensuring correctness.

This change ensures that the function doesn't raise a `KeyError` when an article is missing the `'tags'` key, and it behaves identically to the original function for all inputs.
"""

"""We have this original code:
```
def find_common_tags(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = articles[0]["tags"]
    for article in articles[1:]:
        common_tags = [tag for tag in common_tags if tag in article["tags"]]
    return set(common_tags)
```

We generated optimized code:
```
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        common_tags.intersection_update(article["tags"])
    return common_tags
```

When analyzed the optimized code for correctness with Concolic Testing (using CrossHair DiffBehavior), we found the following counter-example:
```
crosshair diffbehavior --max_uninteresting_iterations 64 code_to_optimize.bubble_sort_typed.find_common_tags code_to_optimize.bubble_sort_typed.find_common_tags2

Given: (articles=[{'tags': ['', '']}, {'tags': ['', '']}, {'tags': []}, {}]), code_to_optimize.bubble_sort_typed.find_common_tags : returns set() code_to_optimize.bubble_sort_typed.find_common_tags2 : raises KeyError()
```

We attempted to fix this with the following candidate:
```python
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()
    
    common_tags = set(articles[0].get("tags", []))
    for article in articles[1:]:
        common_tags.intersection_update(article.get("tags", []))
    return common_tags
```
When analyzing the optimized code for correctness with Concolic Testing (using CrossHair DiffBehavior), we found a different counter-example:
```
Given: (articles=[{'\x00\x00\x00\x00': [], 'tags': ['']}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {'tags': ['']}, {}, {'\x00\x00\x00\x00': [], 'tags': ['']}, {}]),

     code_to_optimize.bubble_sort_typed.find_common_tags : raises KeyError()
  code_to_optimize.bubble_sort_typed.find_common_tags2_1 : returns set()
```

Fix find_common_tags2_1 for correctness relative to find_common_tags, making sure it is still optimized for speed.
"""
"""To fix `find_common_tags2_1` and ensure it behaves identically to `find_common_tags`, we need to address the discrepancy in how the two functions handle the absence of the `'tags'` key and when `common_tags` becomes empty.

**Issue Analysis:**

1. **KeyError Discrepancy:**
   - The original function raises a `KeyError` when accessing `article["tags"]` if the `'tags'` key is missing.
   - Our optimized version used `article.get("tags", [])`, which avoids the `KeyError`, leading to different behavior.

2. **Empty `common_tags`:**
   - In the original function, if `common_tags` becomes empty, the list comprehension doesn't access `article["tags"]`, preventing a `KeyError`.
   - In the optimized version, `intersection_update` still accesses `article["tags"]` even if `common_tags` is empty, leading to unnecessary `KeyError`s.

**Solution:**

- **Mimic Original Exception Behavior:**
  - We should ensure that `find_common_tags2` raises a `KeyError` in the same situations as `find_common_tags`.
  - Avoid accessing `article["tags"]` when `common_tags` is empty, just like the original function.

- **Optimize by Early Exit:**
  - If `common_tags` becomes empty, further intersections won't change it.
  - We can break out of the loop early, improving performance.

**Corrected and Optimized Version:**

```python
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()

    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        if not common_tags:
            break  # No need to continue if common_tags is empty
        common_tags.intersection_update(article["tags"])
    return common_tags
```

**Explanation:**

- **Accessing `article["tags"]`:**
  - We use `article["tags"]` instead of `article.get("tags", [])` to ensure a `KeyError` is raised if `'tags'` is missing, matching the original function's behavior.
  
- **Early Exit on Empty `common_tags`:**
  - The `if not common_tags: break` statement checks if `common_tags` is empty.
  - If it is empty, we break out of the loop, avoiding unnecessary accesses to `article["tags"]` that could raise a `KeyError`.
  
- **Performance Optimization:**
  - By breaking early when `common_tags` is empty, we prevent further iterations and accesses.
  - Using `set` operations with `intersection_update` retains the performance benefits.

**Validation with Counter-Examples:**

- **First Counter-Example:**
  - With `articles=[{'tags': ['', '']}, {'tags': ['', '']}, {'tags': []}, {}]`, the original function returns `set()` without raising `KeyError`.
  - Our corrected `find_common_tags2` also returns `set()` without raising `KeyError` because it avoids accessing `article["tags"]` when `common_tags` is empty.

- **Second Counter-Example:**
  - With an input where an article lacks `'tags'` and `common_tags` is not empty, the original function raises `KeyError`.
  - Our corrected `find_common_tags2` also raises `KeyError` in this situation, matching the original function's behavior.

**Conclusion:**

This corrected version ensures that `find_common_tags2` behaves identically to `find_common_tags` in all cases, raising `KeyError` when the original function does, and avoiding it otherwise. It also maintains the performance optimizations by using set operations and early loop exit.

**Final Optimized Code:**

```python
def find_common_tags2(articles: list[dict[str, list[str]]]) -> set[str]:
    if not articles:
        return set()
        
    common_tags = set(articles[0]["tags"])
    for article in articles[1:]:
        if not common_tags:
            break
        common_tags.intersection_update(article["tags"])
    return common_tags
```"""
