import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codeflash.languages import set_current_language
from codeflash.languages.base import Language
from codeflash.verification.parse_line_profile_test_output import parse_line_profile_results


def test_parse_line_profile_results_non_python_java_json():
    set_current_language(Language.JAVA)

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_file = tmp_path / "Util.java"
        source_file.write_text(
            """public class Util {
    public static int f() {
        int x = 1;
        return x;
    }
}
""",
            encoding="utf-8",
        )
        profile_file = tmp_path / "line_profiler_output.json"
        profile_data = {
            f"{source_file.as_posix()}:3": {
                "hits": 6,
                "time": 1000,
                "file": source_file.as_posix(),
                "line": 3,
                "content": "int x = 1;",
            },
            f"{source_file.as_posix()}:4": {
                "hits": 6,
                "time": 2000,
                "file": source_file.as_posix(),
                "line": 4,
                "content": "return x;",
            },
        }
        profile_file.write_text(json.dumps(profile_data), encoding="utf-8")

        results, _ = parse_line_profile_results(profile_file)

    assert results["unit"] == 1e-9
    assert results["str_out"] == (
        "# Timer unit: 1e-09 s\n"
        "## Function: Util.java\n"
        "## Total time: 3e-06 s\n"
        "|   Hits |   Time |   Per Hit |   % Time | Line Contents   |\n"
        "|-------:|-------:|----------:|---------:|:----------------|\n"
        "|      6 |   1000 |     166.7 |     33.3 | int x = 1;      |\n"
        "|      6 |   2000 |     333.3 |     66.7 | return x;       |\n"
    )
    assert (source_file.as_posix(), 3, "Util.java") in results["timings"]
    assert results["timings"][(source_file.as_posix(), 3, "Util.java")] == [(3, 6, 1000), (4, 6, 2000)]

