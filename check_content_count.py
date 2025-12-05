#!/usr/bin/env python3
"""
Check each line of a .jsonl file and print lines where the string "content"
(as a JSON key) does not occur exactly twice.

Usage:
    python3 scripts/check_content_count.py <path/to/file.jsonl>

Notes:
- By default this counts occurrences of the exact substring '"content"'
  (including quotes) so it matches JSON keys named content and avoids
  accidental matches inside values.
- If you prefer a different matching logic (e.g. count bare word content
  inside values), edit the script accordingly.
"""
import sys
from pathlib import Path


def check_file(path: Path) -> int:
    """Return exit code 0 if all lines ok, 1 if any problematic lines found."""
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 2

    problem_count = 0
    with path.open("r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            # count occurrences of the JSON key name "content"
            cnt = line.count('"content"')
            if cnt != 2:
                problem_count += 1
                # print line number, count and a shortened preview
                preview = line[:300]
                print(f"Line {i}: count={cnt} -- {preview}")

    if problem_count == 0:
        print("All lines contain exactly two occurrences of \"content\".")
        return 0
    else:
        print(f"Found {problem_count} problematic line(s).")
        return 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/check_content_count.py <file.jsonl>")
        sys.exit(2)
    path = Path(sys.argv[1])
    rc = check_file(path)
    sys.exit(rc)
