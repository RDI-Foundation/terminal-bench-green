# Terminal Bench Green

Green (orchestrator) agent for [terminal-bench](https://github.com/laude-institute/terminal-bench-2) evaluations on the [AgentBeats](https://agentbeats.dev) platform.

It downloads tasks from GitHub, spins up Docker environments, sends tasks to a purple agent via A2A, and runs verification to produce scores.

## Architecture

```
AgentBeats Gateway
  → EvalRequest → Green Agent
    → downloads task from terminal-bench-2 repo (Harbor TaskClient)
    → starts Docker environment (Harbor DockerEnvironment)
    → sends prompt + exec URL → Purple Agent (A2A)
      → purple calls POST /exec/{token} to run shell commands
    → runs Harbor Verifier → reward score
    → aggregates results across all tasks
```

### Two servers, one process

`src/server.py` runs two HTTP servers:

| Server | Port | Purpose |
|--------|------|---------|
| A2A server | 9009 | Receives `EvalRequest` from the gateway; speaks A2A protocol |
| Exec API | 9010 | `POST /exec/{session_token}` — hands shell access to purple agents |

## Project Structure

```
src/
├─ server.py      # A2A server + exec endpoint setup
├─ executor.py    # A2A request handling boilerplate
├─ agent.py       # Main orchestration: download, environment, call purple, verify
└─ messenger.py   # A2A client for talking to purple agents
oracle/           # Local oracle purple agent (for testing without amber)
tests/
└─ test_agent.py  # A2A conformance + exec API tests
Dockerfile
pyproject.toml
```

## Request Format

The AgentBeats gateway sends an `EvalRequest` JSON to the A2A server:

```json
{
  "participants": { "agent": "http://purple-agent-url/" },
  "config": {
    "task": "fix-git",
    "oracle": false
  }
}
```

| Config key | Type | Description |
|------------|------|-------------|
| `task` | string | Single task name |
| `tasks` | list or `"all"` | Multiple tasks; `"all"` discovers all tasks from repo root |
| `exclude` | list | Tasks to skip (useful with `tasks: "all"`) |
| `oracle` | bool | Send solve.sh to purple instead of the task description |
| `num_shards` | int | Total number of round-robin shards (default: 1) |
| `shard_index` | int | Which shard this green instance should run (default: 0) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | `/workspace` | Base dir for tasks/trials/repo cache. Must be host-accessible since task containers bind-mount via DooD. |
| `EXEC_BASE_URL` | `http://host.docker.internal:9009` | Base URL sent to purple agents for exec API access. In amber, wired from green's own exec slot. |
| `DOCKER_HOST` | system default | Docker daemon URL. In amber, provided by the amber docker gateway. |

## Running Locally

```bash
# Start green agent (A2A on 9009, exec API on 9010)
WORKSPACE=/tmp/tb-workspace EXEC_BASE_URL=http://127.0.0.1:9010 \
  uv run src/server.py --host 0.0.0.0 --port 9009 --exec-port 9010

# Start oracle purple for local testing (in oracle/ directory)
cd oracle && uv run src/server.py --port 9019

# Run tests
uv run pytest tests/ --agent-url http://localhost:9009
```

## Building the Docker Image

```bash
docker build -t terminal-bench-green:latest .
```

Rebuild whenever `src/` files change before running a scenario.

## Docker-in-Docker Notes

When running inside a container, mount with identical paths so both namespaces agree on the workspace path:

```bash
docker run \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /workspace:/workspace \
  terminal-bench-green:latest
```

## Dependencies

- `harbor>=0.1.44` — task download, Docker environments, verification
- `a2a-sdk[http-server]>=0.3.20` — A2A protocol
- `httpx>=0.28.1` — HTTP client
- `pydantic>=2.12.5` — request validation
- `uvicorn>=0.38.0` — ASGI server
