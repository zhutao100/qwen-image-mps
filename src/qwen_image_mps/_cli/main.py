from __future__ import annotations

import sys

from .commands.edit import edit_image
from .commands.generate import generate_image
from .parser import build_parser


def main() -> None:
    parser = build_parser()

    try:
        if len(sys.argv) > 1 and sys.argv[1] not in [
            "generate",
            "edit",
            "-h",
            "--help",
            "--version",
        ]:
            sys.argv.insert(1, "generate")

        args = parser.parse_args()

        if args.command == "generate":
            list(generate_image(args))
        elif args.command == "edit":
            edit_image(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
