try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies first."
    ) from e

try:
    from .research_librarian_environment import ResearchLibrarianEnvironment
    from ..models import ActionType, ResearchLibrarianObservation
except ImportError:
    from server.research_librarian_environment import ResearchLibrarianEnvironment
    from models import ActionType, ResearchLibrarianObservation


app = create_app(
    ResearchLibrarianEnvironment,
    ActionType,
    ResearchLibrarianObservation,
    env_name="research_librarian",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)