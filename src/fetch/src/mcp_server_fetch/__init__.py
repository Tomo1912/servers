from .server import serve


def main():
    """MCP Fetch Server - HTTP fetching functionality for MCP"""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="give a model the ability to make web requests"
    )
    parser.add_argument("--user-agent", type=str, help="Custom User-Agent string")
    parser.add_argument(
        "--ignore-robots-txt",
        action="store_true",
        help="Ignore robots.txt restrictions",
    )
    parser.add_argument("--proxy-url", type=str, help="Proxy URL to use for requests")
    parser.add_argument(
        "--allow-private-ips",
        action="store_true",
        help="Allow fetching private, loopback and link-local addresses (refused by default)",
    )

    parser.add_argument(
        "--allow-private-host",
        action="append",
        default=[],
        metavar="HOST",
        help="Allow fetching HOST even if it resolves to a private address "
        "(repeatable); redirects from it to other private addresses are still refused",
    )

    args = parser.parse_args()
    asyncio.run(
        serve(
            args.user_agent,
            args.ignore_robots_txt,
            args.proxy_url,
            args.allow_private_ips,
            args.allow_private_host,
        )
    )


if __name__ == "__main__":
    main()
