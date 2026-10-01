#!/usr/bin/env python3
"""Build current HTML and export the two main guides with optional WeasyPrint."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import build_docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=build_docs.ROOT / "output" / "pdf"
    )
    args = parser.parse_args()
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as error:
        raise SystemExit(
            "PDF dependencies are unavailable. See docs/README_docs.md for the "
            "optional installation or use the browser's Print to PDF.\n"
            f"Original error: {error}"
        ) from error

    build_docs.build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Browser-only responsive CSS has no effect on print output.
    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    for stem in build_docs.PDF_STEMS:
        destination = args.output_dir / f"{stem}.pdf"
        HTML(filename=str(build_docs.DOCS / f"{stem}.html")).write_pdf(destination)
        print(f"Built {destination}")


if __name__ == "__main__":
    main()
