"""``torrent-api`` -- serve the HTTP layer against a running chain.

    torrent-api --config configs/local-2shard.yaml

The shard nodes must already be up (``torrent-shard --index N``); this process
holds only a tokenizer and gRPC clients.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="topology YAML")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help=(
            "allow browser requests from this origin; repeatable. Needed for the "
            "dashboard, which runs on a different port than this API. Opt-in "
            "rather than a wildcard default, since a wildcard would let any page "
            "the browser happens to load drive the chain."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    import uvicorn
    from transformers import AutoTokenizer

    from torrent_llm.api import create_app
    from torrent_llm.runner import TopologyConfig

    config = TopologyConfig.from_file(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )
    app = create_app(config, tokenizer)

    if args.cors_origin:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=args.cors_origin,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    print(
        f"serving {config.model_id} over {len(config.nodes)} shards at http://{args.host}:{args.port}"
    )
    print(f"interactive docs at http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
