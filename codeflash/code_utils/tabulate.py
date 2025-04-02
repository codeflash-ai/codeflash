"""Adapted from tabulate (https://github.com/astanin/python-tabulate) written by Sergey Astanin and contributors (MIT License)"""

"""Pretty-print tabular data."""

import warnings
from collections import namedtuple
from collections.abc import Iterable
from itertools import chain, zip_longest as izip_longest
from functools import reduce
import re
import math
import dataclasses
import wcwidth  # optional wide-character (CJK) support

__all__ = ["tabulate", "tabulate_formats"]

# minimum extra space in headers
MIN_PADDING = 2

_DEFAULT_FLOATFMT = "g"
_DEFAULT_INTFMT = ""
_DEFAULT_MISSINGVAL = ""
# default align will be overwritten by "left", "center" or "decimal"
# depending on the formatter
_DEFAULT_ALIGN = "default"


# if True, enable wide-character (CJK) support
WIDE_CHARS_MODE = wcwidth is not None

# Constant that can be used as part of passed rows to generate a separating line
# It is purposely an unprintable character, very unlikely to be used in a table
SEPARATING_LINE = "\001"

Line = namedtuple("Line", ["begin", "hline", "sep", "end"])


DataRow = namedtuple("DataRow", ["begin", "sep", "end"])

TableFormat = namedtuple(
    "TableFormat",
    [
        "lineabove",
        "linebelowheader",
        "linebetweenrows",
        "linebelow",
        "headerrow",
        "datarow",
        "padding",
        "with_header_hide",
    ],
)


def _is_separating_line_value(value):
    return type(value) is str and value.strip() == SEPARATING_LINE


def _is_separating_line(row):
    row_type = type(row)
    is_sl = (row_type == list or row_type == str) and (
        (len(row) >= 1 and _is_separating_line_value(row[0]))
        or (len(row) >= 2 and _is_separating_line_value(row[1]))
    )

    return is_sl


def _pipe_segment_with_colons(align, colwidth):
    """Return a segment of a horizontal line with optional colons which
    indicate column's alignment (as in `pipe` output format)."""
    w = colwidth
    if align in ["right", "decimal"]:
        return ("-" * (w - 1)) + ":"
    elif align == "center":
        return ":" + ("-" * (w - 2)) + ":"
    elif align == "left":
        return ":" + ("-" * (w - 1))
    else:
        return "-" * w


def _pipe_line_with_colons(colwidths, colaligns):
    """Return a horizontal line with optional colons to indicate column's
    alignment (as in `pipe` output format)."""
    if not colaligns:  # e.g. printing an empty data frame (github issue #15)
        colaligns = [""] * len(colwidths)
    segments = [_pipe_segment_with_colons(a, w) for a, w in zip(colaligns, colwidths)]
    return "|" + "|".join(segments) + "|"

_table_formats = {
    "simple": TableFormat(
        lineabove=Line("", "-", "  ", ""),
        linebelowheader=Line("", "-", "  ", ""),
        linebetweenrows=None,
        linebelow=Line("", "-", "  ", ""),
        headerrow=DataRow("", "  ", ""),
        datarow=DataRow("", "  ", ""),
        padding=0,
        with_header_hide=["lineabove", "linebelow"],
    ),
    "pipe": TableFormat(
        lineabove=_pipe_line_with_colons,
        linebelowheader=_pipe_line_with_colons,
        linebetweenrows=None,
        linebelow=None,
        headerrow=DataRow("|", "|", "|"),
        datarow=DataRow("|", "|", "|"),
        padding=1,
        with_header_hide=["lineabove"],
    ),
}

tabulate_formats = list(sorted(_table_formats.keys()))

# The table formats for which multiline cells will be folded into subsequent
# table rows. The key is the original format specified at the API. The value is
# the format that will be used to represent the original format.
multiline_formats = {
    "plain": "plain",
    "pipe": "pipe",

}

_multiline_codes = re.compile(r"\r|\n|\r\n")
_multiline_codes_bytes = re.compile(b"\r|\n|\r\n")

_esc = r"\x1b"
_csi = rf"{_esc}\["
_osc = rf"{_esc}\]"
_st = rf"{_esc}\\"

_ansi_escape_pat = rf"""
    (
        # terminal colors, etc
        {_csi}        # CSI
        [\x30-\x3f]*  # parameter bytes
        [\x20-\x2f]*  # intermediate bytes
        [\x40-\x7e]   # final byte
    |
        # terminal hyperlinks
        {_osc}8;        # OSC opening
        (\w+=\w+:?)*    # key=value params list (submatch 2)
        ;               # delimiter
        ([^{_esc}]+)    # URI - anything but ESC (submatch 3)
        {_st}           # ST
        ([^{_esc}]+)    # link text - anything but ESC (submatch 4)
        {_osc}8;;{_st}  # "closing" OSC sequence
    )
"""
_ansi_codes = re.compile(_ansi_escape_pat, re.VERBOSE)
_ansi_codes_bytes = re.compile(_ansi_escape_pat.encode("utf8"), re.VERBOSE)
_ansi_color_reset_code = "\033[0m"

_float_with_thousands_separators = re.compile(
    r"^(([+-]?[0-9]{1,3})(?:,([0-9]{3}))*)?(?(1)\.[0-9]*|\.[0-9]+)?$"
)

def _isnumber_with_thousands_separator(string):
    try:
        string = string.decode()
    except (UnicodeDecodeError, AttributeError):
        pass

    return bool(re.match(_float_with_thousands_separators, string))


def _isconvertible(conv, string):
    try:
        conv(string)
        return True
    except (ValueError, TypeError):
        return False


def _isnumber(string):
    return (
        # fast path
        type(string) in (float, int)
        # covers 'NaN', +/- 'inf', and eg. '1e2', as well as any type
        # convertible to int/float.
        or (
            _isconvertible(float, string)
            and (
                # some other type convertible to float
                not isinstance(string, (str, bytes))
                # or, a numeric string eg. "1e1...", "NaN", ..., but isn't
                # just an over/underflow
                or (
                    not (math.isinf(float(string)) or math.isnan(float(string)))
                    or string.lower() in ["inf", "-inf", "nan"]
                )
            )
        )
    )


def _isint(string, inttype=int):
    return (
        type(string) is inttype
        or (
            (hasattr(string, "is_integer") or hasattr(string, "__array__"))
            and str(type(string)).startswith("<class 'numpy.int")
        )  # numpy.int64 and similar
        or (
            isinstance(string, (bytes, str)) and _isconvertible(inttype, string)
        )  # integer as string
    )


def _isbool(string):
    return type(string) is bool or (
        isinstance(string, (bytes, str)) and string in ("True", "False")
    )


def _type(string, has_invisible=True, numparse=True):
    if has_invisible and isinstance(string, (str, bytes)):
        string = _strip_ansi(string)

    if string is None or (isinstance(string, (bytes, str)) and not string):
        return type(None)
    elif hasattr(string, "isoformat"):  # datetime.datetime, date, and time
        return str
    elif _isbool(string):
        return bool
    elif numparse and (
        _isint(string)
        or (
            isinstance(string, str)
            and _isnumber_with_thousands_separator(string)
            and "." not in string
        )
    ):
        return int
    elif numparse and (
        _isnumber(string)
        or (isinstance(string, str) and _isnumber_with_thousands_separator(string))
    ):
        return float
    elif isinstance(string, bytes):
        return bytes
    else:
        return str


def _afterpoint(string):
    if _isnumber(string) or _isnumber_with_thousands_separator(string):
        if _isint(string):
            return -1
        else:
            pos = string.rfind(".")
            pos = string.lower().rfind("e") if pos < 0 else pos
            if pos >= 0:
                return len(string) - pos - 1
            else:
                return -1  # no point
    else:
        return -1  # not a number


def _padleft(width, s):
    fmt = "{0:>%ds}" % width
    return fmt.format(s)


def _padright(width, s):
    fmt = "{0:<%ds}" % width
    return fmt.format(s)


def _padboth(width, s):
    fmt = "{0:^%ds}" % width
    return fmt.format(s)


def _padnone(ignore_width, s):
    return s


def _strip_ansi(s):
    if isinstance(s, str):
        return _ansi_codes.sub(r"\4", s)
    else:  # a bytestring
        return _ansi_codes_bytes.sub(r"\4", s)


def _visible_width(s):
    if wcwidth is not None and WIDE_CHARS_MODE:
        len_fn = wcwidth.wcswidth
    else:
        len_fn = len
    if isinstance(s, (str, bytes)):
        return len_fn(_strip_ansi(s))
    else:
        return len_fn(str(s))


def _is_multiline(s):
    if isinstance(s, str):
        return bool(re.search(_multiline_codes, s))
    else:  # a bytestring
        return bool(re.search(_multiline_codes_bytes, s))


def _multiline_width(multiline_s, line_width_fn=len):
    return max(map(line_width_fn, re.split("[\r\n]", multiline_s)))


def _choose_width_fn(has_invisible, enable_widechars, is_multiline):
    if has_invisible:
        line_width_fn = _visible_width
    elif enable_widechars:  # optional wide-character support if available
        line_width_fn = wcwidth.wcswidth
    else:
        line_width_fn = len
    if is_multiline:
        width_fn = lambda s: _multiline_width(s, line_width_fn)  # noqa
    else:
        width_fn = line_width_fn
    return width_fn


def _align_column_choose_padfn(strings, alignment, has_invisible, preserve_whitespace):
    if alignment == "right":
        if not preserve_whitespace:
            strings = [s.strip() for s in strings]
        padfn = _padleft
    elif alignment == "center":
        if not preserve_whitespace:
            strings = [s.strip() for s in strings]
        padfn = _padboth
    elif alignment == "decimal":
        if has_invisible:
            decimals = [_afterpoint(_strip_ansi(s)) for s in strings]
        else:
            decimals = [_afterpoint(s) for s in strings]
        maxdecimals = max(decimals)
        strings = [s + (maxdecimals - decs) * " " for s, decs in zip(strings, decimals)]
        padfn = _padleft
    elif not alignment:
        padfn = _padnone
    else:
        if not preserve_whitespace:
            strings = [s.strip() for s in strings]
        padfn = _padright
    return strings, padfn


def _align_column_choose_width_fn(has_invisible, enable_widechars, is_multiline):
    if has_invisible:
        line_width_fn = _visible_width
    elif enable_widechars:  # optional wide-character support if available
        line_width_fn = wcwidth.wcswidth
    else:
        line_width_fn = len
    if is_multiline:
        width_fn = lambda s: _align_column_multiline_width(s, line_width_fn)  # noqa
    else:
        width_fn = line_width_fn
    return width_fn


def _align_column_multiline_width(multiline_s, line_width_fn=len):
    return list(map(line_width_fn, re.split("[\r\n]", multiline_s)))


def _flat_list(nested_list):
    ret = []
    for item in nested_list:
        if isinstance(item, list):
            ret.extend(item)
        else:
            ret.append(item)
    return ret


def _align_column(
    strings,
    alignment,
    minwidth=0,
    has_invisible=True,
    enable_widechars=False,
    is_multiline=False,
    preserve_whitespace=False,
):
    strings, padfn = _align_column_choose_padfn(
        strings, alignment, has_invisible, preserve_whitespace
    )
    width_fn = _align_column_choose_width_fn(
        has_invisible, enable_widechars, is_multiline
    )

    s_widths = list(map(width_fn, strings))
    maxwidth = max(max(_flat_list(s_widths)), minwidth)
    # TODO: refactor column alignment in single-line and multiline modes
    if is_multiline:
        if not enable_widechars and not has_invisible:
            padded_strings = [
                "\n".join([padfn(maxwidth, s) for s in ms.splitlines()])
                for ms in strings
            ]
        else:
            # enable wide-character width corrections
            s_lens = [[len(s) for s in re.split("[\r\n]", ms)] for ms in strings]
            visible_widths = [
                [maxwidth - (w - l) for w, l in zip(mw, ml)]
                for mw, ml in zip(s_widths, s_lens)
            ]
            # wcswidth and _visible_width don't count invisible characters;
            # padfn doesn't need to apply another correction
            padded_strings = [
                "\n".join([padfn(w, s) for s, w in zip((ms.splitlines() or ms), mw)])
                for ms, mw in zip(strings, visible_widths)
            ]
    else:  # single-line cell values
        if not enable_widechars and not has_invisible:
            padded_strings = [padfn(maxwidth, s) for s in strings]
        else:
            # enable wide-character width corrections
            s_lens = list(map(len, strings))
            visible_widths = [maxwidth - (w - l) for w, l in zip(s_widths, s_lens)]
            # wcswidth and _visible_width don't count invisible characters;
            # padfn doesn't need to apply another correction
            padded_strings = [padfn(w, s) for s, w in zip(strings, visible_widths)]
    return padded_strings


def _more_generic(type1, type2):
    types = {
        type(None): 0,
        bool: 1,
        int: 2,
        float: 3,
        bytes: 4,
        str: 5,
    }
    invtypes = {
        5: str,
        4: bytes,
        3: float,
        2: int,
        1: bool,
        0: type(None),
    }
    moregeneric = max(types.get(type1, 5), types.get(type2, 5))
    return invtypes[moregeneric]


def _column_type(strings, has_invisible=True, numparse=True):
    types = [_type(s, has_invisible, numparse) for s in strings]
    return reduce(_more_generic, types, bool)


def _format(val, valtype, floatfmt, intfmt, missingval="", has_invisible=True):
    if val is None:
        return missingval
    if isinstance(val, (bytes, str)) and not val:
        return ""

    if valtype is str:
        return f"{val}"
    elif valtype is int:
        if isinstance(val, str):
            val_striped = val.encode("unicode_escape").decode("utf-8")
            colored = re.search(
                r"(\\[xX]+[0-9a-fA-F]+\[\d+[mM]+)([0-9.]+)(\\.*)$", val_striped
            )
            if colored:
                total_groups = len(colored.groups())
                if total_groups == 3:
                    digits = colored.group(2)
                    if digits.isdigit():
                        val_new = (
                            colored.group(1)
                            + format(int(digits), intfmt)
                            + colored.group(3)
                        )
                        val = val_new.encode("utf-8").decode("unicode_escape")
            intfmt = ""
        return format(val, intfmt)
    elif valtype is bytes:
        try:
            return str(val, "ascii")
        except (TypeError, UnicodeDecodeError):
            return str(val)
    elif valtype is float:
        is_a_colored_number = has_invisible and isinstance(val, (str, bytes))
        if is_a_colored_number:
            raw_val = _strip_ansi(val)
            formatted_val = format(float(raw_val), floatfmt)
            return val.replace(raw_val, formatted_val)
        else:
            if isinstance(val, str) and "," in val:
                val = val.replace(",", "")  # handle thousands-separators
            return format(float(val), floatfmt)
    else:
        return f"{val}"


def _align_header(
    header, alignment, width, visible_width, is_multiline=False, width_fn=None
):
    "Pad string header to width chars given known visible_width of the header."
    if is_multiline:
        header_lines = re.split(_multiline_codes, header)
        padded_lines = [
            _align_header(h, alignment, width, width_fn(h)) for h in header_lines
        ]
        return "\n".join(padded_lines)
    # else: not multiline
    ninvisible = len(header) - visible_width
    width += ninvisible
    if alignment == "left":
        return _padright(width, header)
    elif alignment == "center":
        return _padboth(width, header)
    elif not alignment:
        return f"{header}"
    else:
        return _padleft(width, header)


def _remove_separating_lines(rows):
    if isinstance(rows, list):
        separating_lines = []
        sans_rows = []
        for index, row in enumerate(rows):
            if _is_separating_line(row):
                separating_lines.append(index)
            else:
                sans_rows.append(row)
        return sans_rows, separating_lines
    else:
        return rows, None

def _bool(val):
    "A wrapper around standard bool() which doesn't throw on NumPy arrays"
    try:
        return bool(val)
    except ValueError:  # val is likely to be a numpy array with many elements
        return False


def _normalize_tabular_data(tabular_data, headers, showindex="default"):
    try:
        bool(headers)
    except ValueError:  # numpy.ndarray, pandas.core.index.Index, ...
        headers = list(headers)

    err_msg = (
        "\n\nTo build a table python-tabulate requires two-dimensional data "
        "like a list of lists or similar."
        "\nDid you forget a pair of extra [] or ',' in ()?"
    )
    index = None
    if hasattr(tabular_data, "keys") and hasattr(tabular_data, "values"):
        # dict-like and pandas.DataFrame?
        if hasattr(tabular_data.values, "__call__"):
            # likely a conventional dict
            keys = tabular_data.keys()
            try:
                rows = list(
                    izip_longest(*tabular_data.values())
                )  # columns have to be transposed
            except TypeError:  # not iterable
                raise TypeError(err_msg)

        elif hasattr(tabular_data, "index"):
            # values is a property, has .index => it's likely a pandas.DataFrame (pandas 0.11.0)
            keys = list(tabular_data)
            if (
                showindex in ["default", "always", True]
                and tabular_data.index.name is not None
            ):
                if isinstance(tabular_data.index.name, list):
                    keys[:0] = tabular_data.index.name
                else:
                    keys[:0] = [tabular_data.index.name]
            vals = tabular_data.values  # values matrix doesn't need to be transposed
            # for DataFrames add an index per default
            index = list(tabular_data.index)
            rows = [list(row) for row in vals]
        else:
            raise ValueError("tabular data doesn't appear to be a dict or a DataFrame")

        if headers == "keys":
            headers = list(map(str, keys))  # headers should be strings

    else:  # it's a usual iterable of iterables, or a NumPy array, or an iterable of dataclasses
        try:
            rows = list(tabular_data)
        except TypeError:  # not iterable
            raise TypeError(err_msg)

        if headers == "keys" and not rows:
            # an empty table (issue #81)
            headers = []
        elif (
            headers == "keys"
            and hasattr(tabular_data, "dtype")
            and getattr(tabular_data.dtype, "names")
        ):
            # numpy record array
            headers = tabular_data.dtype.names
        elif (
            headers == "keys"
            and len(rows) > 0
            and isinstance(rows[0], tuple)
            and hasattr(rows[0], "_fields")
        ):
            # namedtuple
            headers = list(map(str, rows[0]._fields))
        elif len(rows) > 0 and hasattr(rows[0], "keys") and hasattr(rows[0], "values"):
            # dict-like object
            uniq_keys = set()  # implements hashed lookup
            keys = []  # storage for set
            if headers == "firstrow":
                firstdict = rows[0] if len(rows) > 0 else {}
                keys.extend(firstdict.keys())
                uniq_keys.update(keys)
                rows = rows[1:]
            for row in rows:
                for k in row.keys():
                    # Save unique items in input order
                    if k not in uniq_keys:
                        keys.append(k)
                        uniq_keys.add(k)
            if headers == "keys":
                headers = keys
            elif isinstance(headers, dict):
                # a dict of headers for a list of dicts
                headers = [headers.get(k, k) for k in keys]
                headers = list(map(str, headers))
            elif headers == "firstrow":
                if len(rows) > 0:
                    headers = [firstdict.get(k, k) for k in keys]
                    headers = list(map(str, headers))
                else:
                    headers = []
            elif headers:
                raise ValueError(
                    "headers for a list of dicts is not a dict or a keyword"
                )
            rows = [[row.get(k) for k in keys] for row in rows]

        elif (
            headers == "keys"
            and hasattr(tabular_data, "description")
            and hasattr(tabular_data, "fetchone")
            and hasattr(tabular_data, "rowcount")
        ):
            # Python Database API cursor object (PEP 0249)
            # print tabulate(cursor, headers='keys')
            headers = [column[0] for column in tabular_data.description]

        elif (
            dataclasses is not None
            and len(rows) > 0
            and dataclasses.is_dataclass(rows[0])
        ):
            # Python's dataclass
            field_names = [field.name for field in dataclasses.fields(rows[0])]
            if headers == "keys":
                headers = field_names
            rows = [[getattr(row, f) for f in field_names] for row in rows]

        elif headers == "keys" and len(rows) > 0:
            # keys are column indices
            headers = list(map(str, range(len(rows[0]))))

    # take headers from the first row if necessary
    if headers == "firstrow" and len(rows) > 0:
        if index is not None:
            headers = [index[0]] + list(rows[0])
            index = index[1:]
        else:
            headers = rows[0]
        headers = list(map(str, headers))  # headers should be strings
        rows = rows[1:]
    elif headers == "firstrow":
        headers = []

    headers = list(map(str, headers))
    #    rows = list(map(list, rows))
    rows = list(map(lambda r: r if _is_separating_line(r) else list(r), rows))

    # add or remove an index column
    showindex_is_a_str = type(showindex) in [str, bytes]
    if showindex == "never" or (not _bool(showindex) and not showindex_is_a_str):
        pass

    # pad with empty headers for initial columns if necessary
    headers_pad = 0
    if headers and len(rows) > 0:
        headers_pad = max(0, len(rows[0]) - len(headers))
        headers = [""] * headers_pad + headers

    return rows, headers, headers_pad

def _to_str(s, encoding="utf8", errors="ignore"):
    if isinstance(s, bytes):
        return s.decode(encoding=encoding, errors=errors)
    return str(s)


def tabulate(
    tabular_data,
    headers=(),
    tablefmt="simple",
    floatfmt=_DEFAULT_FLOATFMT,
    intfmt=_DEFAULT_INTFMT,
    numalign=_DEFAULT_ALIGN,
    stralign=_DEFAULT_ALIGN,
    missingval=_DEFAULT_MISSINGVAL,
    showindex="default",
    disable_numparse=False,
    colglobalalign=None,
    colalign=None,
    preserve_whitespace=False,
    maxcolwidths=None,
    headersglobalalign=None,
    headersalign=None,
    rowalign=None,
    maxheadercolwidths=None,
):
    if tabular_data is None:
        tabular_data = []

    list_of_lists, headers, headers_pad = _normalize_tabular_data(
        tabular_data, headers, showindex=showindex
    )
    list_of_lists, separating_lines = _remove_separating_lines(list_of_lists)

    # PrettyTable formatting does not use any extra padding.
    # Numbers are not parsed and are treated the same as strings for alignment.
    # Check if pretty is the format being used and override the defaults so it
    # does not impact other formats.
    min_padding = MIN_PADDING
    if tablefmt == "pretty":
        min_padding = 0
        disable_numparse = True
        numalign = "center" if numalign == _DEFAULT_ALIGN else numalign
        stralign = "center" if stralign == _DEFAULT_ALIGN else stralign
    else:
        numalign = "decimal" if numalign == _DEFAULT_ALIGN else numalign
        stralign = "left" if stralign == _DEFAULT_ALIGN else stralign

    # 'colon_grid' uses colons in the line beneath the header to represent a column's
    # alignment instead of literally aligning the text differently. Hence,
    # left alignment of the data in the text output is enforced.
    if tablefmt == "colon_grid":
        colglobalalign = "left"
        headersglobalalign = "left"

    # optimization: look for ANSI control codes once,
    # enable smart width functions only if a control code is found
    #
    # convert the headers and rows into a single, tab-delimited string ensuring
    # that any bytestrings are decoded safely (i.e. errors ignored)
    plain_text = "\t".join(
        chain(
            # headers
            map(_to_str, headers),
            # rows: chain the rows together into a single iterable after mapping
            # the bytestring conversino to each cell value
            chain.from_iterable(map(_to_str, row) for row in list_of_lists),
        )
    )

    has_invisible = _ansi_codes.search(plain_text) is not None

    enable_widechars = wcwidth is not None and WIDE_CHARS_MODE
    if (
        not isinstance(tablefmt, TableFormat)
        and tablefmt in multiline_formats
        and _is_multiline(plain_text)
    ):
        tablefmt = multiline_formats.get(tablefmt, tablefmt)
        is_multiline = True
    else:
        is_multiline = False
    width_fn = _choose_width_fn(has_invisible, enable_widechars, is_multiline)

    # format rows and columns, convert numeric values to strings
    cols = list(izip_longest(*list_of_lists))
    numparses = _expand_numparse(disable_numparse, len(cols))
    coltypes = [_column_type(col, numparse=np) for col, np in zip(cols, numparses)]
    if isinstance(floatfmt, str):  # old version
        float_formats = len(cols) * [
            floatfmt
        ]  # just duplicate the string to use in each column
    else:  # if floatfmt is list, tuple etc we have one per column
        float_formats = list(floatfmt)
        if len(float_formats) < len(cols):
            float_formats.extend((len(cols) - len(float_formats)) * [_DEFAULT_FLOATFMT])
    if isinstance(intfmt, str):  # old version
        int_formats = len(cols) * [
            intfmt
        ]  # just duplicate the string to use in each column
    else:  # if intfmt is list, tuple etc we have one per column
        int_formats = list(intfmt)
        if len(int_formats) < len(cols):
            int_formats.extend((len(cols) - len(int_formats)) * [_DEFAULT_INTFMT])
    if isinstance(missingval, str):
        missing_vals = len(cols) * [missingval]
    else:
        missing_vals = list(missingval)
        if len(missing_vals) < len(cols):
            missing_vals.extend((len(cols) - len(missing_vals)) * [_DEFAULT_MISSINGVAL])
    cols = [
        [_format(v, ct, fl_fmt, int_fmt, miss_v, has_invisible) for v in c]
        for c, ct, fl_fmt, int_fmt, miss_v in zip(
            cols, coltypes, float_formats, int_formats, missing_vals
        )
    ]

    # align columns
    # first set global alignment
    if colglobalalign is not None:  # if global alignment provided
        aligns = [colglobalalign] * len(cols)
    else:  # default
        aligns = [numalign if ct in [int, float] else stralign for ct in coltypes]
    # then specific alignments
    if colalign is not None:
        assert isinstance(colalign, Iterable)
        if isinstance(colalign, str):
            warnings.warn(
                f"As a string, `colalign` is interpreted as {[c for c in colalign]}. "
                f'Did you mean `colglobalalign = "{colalign}"` or `colalign = ("{colalign}",)`?',
                stacklevel=2,
            )
        for idx, align in enumerate(colalign):
            if not idx < len(aligns):
                break
            elif align != "global":
                aligns[idx] = align
    minwidths = (
        [width_fn(h) + min_padding for h in headers] if headers else [0] * len(cols)
    )
    aligns_copy = aligns.copy()
    # Reset alignments in copy of alignments list to "left" for 'colon_grid' format,
    # which enforces left alignment in the text output of the data.
    if tablefmt == "colon_grid":
        aligns_copy = ["left"] * len(cols)
    cols = [
        _align_column(
            c,
            a,
            minw,
            has_invisible,
            enable_widechars,
            is_multiline,
            preserve_whitespace,
        )
        for c, a, minw in zip(cols, aligns_copy, minwidths)
    ]

    aligns_headers = None
    if headers:
        # align headers and add headers
        t_cols = cols or [[""]] * len(headers)
        # first set global alignment
        if headersglobalalign is not None:  # if global alignment provided
            aligns_headers = [headersglobalalign] * len(t_cols)
        else:  # default
            aligns_headers = aligns or [stralign] * len(headers)
        # then specific header alignments
        if headersalign is not None:
            assert isinstance(headersalign, Iterable)
            if isinstance(headersalign, str):
                warnings.warn(
                    f"As a string, `headersalign` is interpreted as {[c for c in headersalign]}. "
                    f'Did you mean `headersglobalalign = "{headersalign}"` '
                    f'or `headersalign = ("{headersalign}",)`?',
                    stacklevel=2,
                )
            for idx, align in enumerate(headersalign):
                hidx = headers_pad + idx
                if not hidx < len(aligns_headers):
                    break
                elif align == "same" and hidx < len(aligns):  # same as column align
                    aligns_headers[hidx] = aligns[hidx]
                elif align != "global":
                    aligns_headers[hidx] = align
        minwidths = [
            max(minw, max(width_fn(cl) for cl in c))
            for minw, c in zip(minwidths, t_cols)
        ]
        headers = [
            _align_header(h, a, minw, width_fn(h), is_multiline, width_fn)
            for h, a, minw in zip(headers, aligns_headers, minwidths)
        ]
        rows = list(zip(*cols))
    else:
        minwidths = [max(width_fn(cl) for cl in c) for c in cols]
        rows = list(zip(*cols))

    if not isinstance(tablefmt, TableFormat):
        tablefmt = _table_formats.get(tablefmt, _table_formats["simple"])

    ra_default = rowalign if isinstance(rowalign, str) else None
    rowaligns = _expand_iterable(rowalign, len(rows), ra_default)
    return _format_table(
        tablefmt,
        headers,
        aligns_headers,
        rows,
        minwidths,
        aligns,
        is_multiline,
        rowaligns=rowaligns,
    )


def _expand_numparse(disable_numparse, column_count):
    if isinstance(disable_numparse, Iterable):
        numparses = [True] * column_count
        for index in disable_numparse:
            numparses[index] = False
        return numparses
    else:
        return [not disable_numparse] * column_count


def _expand_iterable(original, num_desired, default):
    if isinstance(original, Iterable) and not isinstance(original, str):
        return original + [default] * (num_desired - len(original))
    else:
        return [default] * num_desired


def _pad_row(cells, padding):
    if cells:
        if cells == SEPARATING_LINE:
            return SEPARATING_LINE
        pad = " " * padding
        padded_cells = [pad + cell + pad for cell in cells]
        return padded_cells
    else:
        return cells


def _build_simple_row(padded_cells, rowfmt):
    begin, sep, end = rowfmt
    return (begin + sep.join(padded_cells) + end).rstrip()


def _build_row(padded_cells, colwidths, colaligns, rowfmt):
    if not rowfmt:
        return None
    if hasattr(rowfmt, "__call__"):
        return rowfmt(padded_cells, colwidths, colaligns)
    else:
        return _build_simple_row(padded_cells, rowfmt)

def _append_basic_row(lines, padded_cells, colwidths, colaligns, rowfmt, rowalign=None):
    # NOTE: rowalign is ignored and exists for api compatibility with _append_multiline_row
    lines.append(_build_row(padded_cells, colwidths, colaligns, rowfmt))
    return lines

def _build_line(colwidths, colaligns, linefmt):
    "Return a string which represents a horizontal line."
    if not linefmt:
        return None
    if hasattr(linefmt, "__call__"):
        return linefmt(colwidths, colaligns)
    else:
        begin, fill, sep, end = linefmt
        cells = [fill * w for w in colwidths]
        return _build_simple_row(cells, (begin, sep, end))


def _append_line(lines, colwidths, colaligns, linefmt):
    lines.append(_build_line(colwidths, colaligns, linefmt))
    return lines

def _format_table(
    fmt, headers, headersaligns, rows, colwidths, colaligns, is_multiline, rowaligns
):
    lines = []
    hidden = fmt.with_header_hide if (headers and fmt.with_header_hide) else []
    pad = fmt.padding
    headerrow = fmt.headerrow

    padded_widths = [(w + 2 * pad) for w in colwidths]
    pad_row = _pad_row
    append_row = _append_basic_row

    padded_headers = pad_row(headers, pad)

    if fmt.lineabove and "lineabove" not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.lineabove)

    if padded_headers:
        append_row(lines, padded_headers, padded_widths, headersaligns, headerrow)
        if fmt.linebelowheader and "linebelowheader" not in hidden:
            _append_line(lines, padded_widths, colaligns, fmt.linebelowheader)

    if rows and fmt.linebetweenrows and "linebetweenrows" not in hidden:
        # initial rows with a line below
        for row, ralign in zip(rows[:-1], rowaligns):
            if row != SEPARATING_LINE:
                append_row(
                    lines,
                    pad_row(row, pad),
                    padded_widths,
                    colaligns,
                    fmt.datarow,
                    rowalign=ralign,
                )
            _append_line(lines, padded_widths, colaligns, fmt.linebetweenrows)
        # the last row without a line below
        append_row(
            lines,
            pad_row(rows[-1], pad),
            padded_widths,
            colaligns,
            fmt.datarow,
            rowalign=rowaligns[-1],
        )
    else:
        separating_line = (
            fmt.linebetweenrows
            or fmt.linebelowheader
            or fmt.linebelow
            or fmt.lineabove
            or Line("", "", "", "")
        )
        for row in rows:
            # test to see if either the 1st column or the 2nd column (account for showindex) has
            # the SEPARATING_LINE flag
            if _is_separating_line(row):
                _append_line(lines, padded_widths, colaligns, separating_line)
            else:
                append_row(
                    lines, pad_row(row, pad), padded_widths, colaligns, fmt.datarow
                )

    if fmt.linebelow and "linebelow" not in hidden:
        _append_line(lines, padded_widths, colaligns, fmt.linebelow)

    if headers or rows:
        output = "\n".join(lines)
        return output
    else:  # a completely empty table
        return ""