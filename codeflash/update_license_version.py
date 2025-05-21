import re
from datetime import datetime
from pathlib import Path

from version import __version_tuple__


def main() -> None:
    # Use the version tuple from version.py
    version = __version_tuple__

    # Use the major and minor version components from the version tuple
    major_minor_version = ".".join(map(str, version[:2]))

    # Define the pattern to search for and the replacement string for the version
    version_pattern = re.compile(r"(Licensed Work:\s+Codeflash Client version\s+)(0\.\d+)(\.x)")
    version_replacement = r"\g<1>" + major_minor_version + r".x"

    # Read the LICENSE file
    with (Path(__file__).parent / "LICENSE").open(encoding="utf8") as file:
        license_text = file.read()

    # Replace the version in the LICENSE file
    updated_license_text = version_pattern.sub(version_replacement, license_text)

    # Extract the current version from the LICENSE file
    current_version_match = re.search(r"version\s+(\d+\.\d+)\.x", license_text)
    if current_version_match:
        current_major_minor_version = current_version_match.group(1)
        # Check if the minor version has changed and update the date if necessary
        if current_major_minor_version and major_minor_version != current_major_minor_version:
            # Calculate the new date, which is the current year plus four years
            new_year = datetime.now().year + 4  # noqa: DTZ005
            new_date = f"{new_year}-{datetime.now().strftime('%m-%d')}"  # noqa: DTZ005
            # Define the pattern to search for and the replacement string for the date
            date_pattern = re.compile(r"(Change Date:\s+)(\d{4}-\d{2}-\d{2})")
            date_replacement = r"\g<1>" + new_date
            updated_license_text = date_pattern.sub(date_replacement, updated_license_text)

    # Write the updated LICENSE file
    with (Path(__file__).parent / "LICENSE").open("w", encoding="utf8") as file:
        file.write(updated_license_text)


if __name__ == "__main__":
    main()
