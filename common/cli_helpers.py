#!/usr/bin/env python3
"""Shared argparse helpers used across the monolayer/bilayer generation CLIs."""


def add_mp_args(parser):
    """
    Add the standard Materials Project lookup flags to an argparse parser:
    --no-mp, --mp-api-key, --mp-refresh, --mp-verbose, --strict-validation.

    Populates args.no_mp, args.mp_api_key, args.mp_refresh, args.mp_verbose,
    args.strict_validation.
    """
    parser.add_argument(
        "--no-mp",
        action="store_true",
        help="Disable Materials Project structure lookup and use template-only generation",
    )
    parser.add_argument(
        "--mp-api-key",
        type=str,
        default=None,
        help="Materials Project API key (default: MP_API_KEY environment variable)",
    )
    parser.add_argument(
        "--mp-refresh",
        action="store_true",
        help="Refresh Materials Project structure cache for queried materials",
    )
    parser.add_argument(
        "--mp-verbose",
        action="store_true",
        help="Print MP selection/fallback details",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat MP structure validation failures as hard errors",
    )
    return parser
