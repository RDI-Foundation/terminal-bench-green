import argparse

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from harbor.environments.base import BaseEnvironment
from executor import Executor


# Shared exec session registry: session_token -> BaseEnvironment
exec_sessions: dict[str, BaseEnvironment] = {}


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port for the A2A server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="terminal-bench",
        name="Terminal Bench",
        description="Evaluates AI agents on terminal-bench 2.0 tasks via the A2A shell protocol and running verification.",
        tags=["evaluation", "benchmark", "terminal"],
        examples=[],
    )

    agent_card = AgentCard(
        name="Terminal Bench Green",
        description="Green agent that orchestrates terminal-bench 2.0 benchmark evaluations. Manages Docker containers, provides shell access to purple agents, and runs verification to produce scores.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(exec_sessions),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
