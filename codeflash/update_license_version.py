import re

from version import __version_tuple__


def main():
    # Use the version tuple from version.py
    version = __version_tuple__

    # Use the major and minor version components from the version tuple
    major_minor_version = ".".join(map(str, version[:2]))

    # Define the pattern to search for and the replacement string
    pattern = re.compile(r"(Licensed Work:\s+CodeFlash Client version\s+)(0\.\d+)(\.x)")
    replacement = r"\g<1>" + major_minor_version + r".x"

    # Read the LICENSE file
    with open("codeflash/LICENSE", "r") as file:
        license_text = file.read()

    # Replace the version in the LICENSE file
    updated_license_text = pattern.sub(replacement, license_text)

    # Write the updated LICENSE file
    with open("codeflash/LICENSE", "w") as file:
        file.write(updated_license_text)


if __name__ == "__main__":
    main()
