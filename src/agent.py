import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from harbor.tasks.client import TaskClient
from harbor.models.task.task import Task
from harbor.models.task.id import LocalTaskId
from harbor.environments.base import BaseEnvironment
from dood_environment import DoodDockerEnvironment
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import (
    AddTestsDirError,
    DownloadVerifierDirError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Verifier,
    VerifierOutputParseError,
)

from messenger import Messenger

logger = logging.getLogger(__name__)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
TASKS_DIR = WORKSPACE / "tasks"
TRIALS_DIR = WORKSPACE / "trials"

_PREBAKED_REPO = Path("/home/agent/terminal-bench-2")
TASK_REPO_DIR = WORKSPACE / "repo"


def _ensure_task_repo() -> None:
    """Copy the pre-baked task repo into WORKSPACE so paths are visible to the
    host Docker daemon (required for DOOD bind-mounts)."""
    if TASK_REPO_DIR.exists():
        return
    import shutil
    shutil.copytree(_PREBAKED_REPO, TASK_REPO_DIR)


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


def _list_all_tasks() -> list[str]:
    """List all task directories from the local repo."""
    return sorted(
        p.name for p in TASK_REPO_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _get_task_names(config: dict[str, Any]) -> list[str]:
    """Extract task name(s) from config. Supports 'task' (single), 'tasks' (list or 'all'), and optional 'exclude' list."""
    if "tasks" in config:
        tasks = config["tasks"]
        if tasks == "all":
            names = _list_all_tasks()
        elif isinstance(tasks, list):
            names = tasks
        else:
            raise ValueError("'tasks' must be a list of task names or 'all'")
    elif "task" in config:
        names = [config["task"]]
    else:
        raise ValueError("Config must contain 'task' or 'tasks'")

    exclude = config.get("exclude", [])
    if exclude:
        exclude_set = set(exclude)
        names = [n for n in names if n not in exclude_set]
    return names


def _get_shard_task_names(task_names: list[str], config: dict[str, Any]) -> list[str]:
    """Select this shard's tasks using round-robin slicing."""
    num_shards = int(config.get("num_shards", 1))
    shard_index = int(config.get("shard_index", 0))
    if num_shards < 1:
        raise ValueError("'num_shards' must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("'shard_index' must be in [0, num_shards)")
    return task_names[shard_index::num_shards]


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept direct task config or unwrap a nested assessment_config object."""
    if "task" in config or "tasks" in config:
        return config

    nested = config.get("assessment_config")
    if isinstance(nested, dict) and ("task" in nested or "tasks" in nested):
        return nested

    return config


def _missing_task_config_message(config: dict[str, Any]) -> str:
    keys = ", ".join(sorted(config.keys())) or "(none)"
    return (
        "Config must contain 'task' or 'tasks' at the top level or under "
        f"'assessment_config'; received keys: {keys}"
    )


def _read_text_tail(path: Path, max_chars: int = 4000) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(errors="replace").strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return f"...\n{text[-max_chars:]}"


def _format_verifier_failure(
    task_name: str,
    error: Exception,
    trial_paths: TrialPaths,
) -> str:
    message = f"Verifier failed for {task_name}: {error}"
    test_output = _read_text_tail(trial_paths.test_stdout_path)
    if test_output:
        return f"{message}\n\ntest.sh output:\n{test_output}"
    return message


class Agent:
    required_roles: list[str] = ["agent"]
    # The amber docker gateway scopes task containers at the green-component
    # level, not the individual task level. Under DOOD, concurrent task
    # environments can therefore interfere with each other during compose up/down.
    # Keep exactly one task environment alive at a time.
    _docker_lock = asyncio.Lock()

    def __init__(self, exec_sessions: dict[str, BaseEnvironment]):
        self.messenger = Messenger()
        self.task_client = TaskClient()
        self.exec_sessions = exec_sessions

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        config = _normalize_config(request.config)
        if "task" not in config and "tasks" not in config:
            return False, _missing_task_config_message(request.config)
        exclude = config.get("exclude", [])
        if not isinstance(exclude, list):
            return False, "'exclude' must be a list of task names"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        _ensure_task_repo()
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        config = _normalize_config(request.config)
        task_names = _get_task_names(config)
        shard_task_names = _get_shard_task_names(task_names, config)
        agent_url = str(request.participants["agent"])
        oracle = config.get("oracle", False)
        num_shards = int(config.get("num_shards", 1))
        shard_index = int(config.get("shard_index", 0))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation of {len(shard_task_names)} task(s) on shard "
                f"{shard_index + 1}/{num_shards}{' (oracle)' if oracle else ''}"
            ),
        )

        task_rewards: dict[str, Any] = {}

        async def run_task(task_num: int, task_name: str) -> None:
            tag = f"[{task_num}/{len(shard_task_names)}]"
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} Starting task: {task_name}"),
            )
            try:
                rewards = await self._run_single_task(task_name, agent_url, updater, oracle=oracle, tag=tag)
                task_rewards[task_name] = rewards
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{tag} Completed task: {task_name} — {rewards}"),
                )
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}", exc_info=True)
                task_rewards[task_name] = {"error": str(e)}
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{tag} Failed task: {task_name} — {e}"),
                )

        for i, name in enumerate(shard_task_names, start=1):
            await run_task(i, name)

        # Aggregate results
        total_score = sum(
            r.get("reward", 0)
            for r in task_rewards.values()
            if isinstance(r, dict) and "error" not in r
        )
        num_tasks = len(shard_task_names)
        num_passed = sum(
            1 for r in task_rewards.values()
            if isinstance(r, dict) and "error" not in r and r.get("reward", 0) > 0
        )

        summary = f"Completed {num_tasks} task(s): {num_passed} passed, score {total_score}/{num_tasks}"

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary)),
                Part(root=DataPart(data={
                    "score": total_score,
                    "max_score": num_tasks,
                    "pass_rate": (num_passed / num_tasks * 100) if num_tasks > 0 else 0,
                    "task_rewards": task_rewards,
                })),
            ],
            name="Result",
        )

    async def _run_single_task(
        self, task_name: str, agent_url: str, updater: TaskUpdater,
        *, oracle: bool = False, tag: str = "",
    ) -> dict:
        async with self._docker_lock:
            return await self._run_single_task_locked(
                task_name,
                agent_url,
                updater,
                oracle=oracle,
                tag=tag,
            )

    async def _run_single_task_locked(
        self, task_name: str, agent_url: str, updater: TaskUpdater,
        *, oracle: bool = False, tag: str = "",
    ) -> dict:
        """Run a single task. Returns the rewards dict."""
        session_token = uuid4().hex
        session_id = uuid4().hex
        environment: BaseEnvironment | None = None

        try:
            # 1. Load task from pre-baked local repo.
            task_id = LocalTaskId(path=TASK_REPO_DIR / task_name)
            batch = await self.task_client.download_tasks(
                [task_id], output_dir=TASKS_DIR
            )
            task = Task(batch.paths[0])

            # 2. Create trial paths
            trial_dir = TRIALS_DIR / session_id
            trial_paths = TrialPaths(trial_dir=trial_dir)
            trial_paths.mkdir()

            # 3. Create and start environment
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Starting environment..."),
            )
            environment = DoodDockerEnvironment(
                environment_dir=task.paths.environment_dir,
                environment_name=f"terminal-bench-{task_name}",
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task.config.environment,
                logger=logger,
            )
            build_timeout = task.config.environment.build_timeout_sec
            await asyncio.wait_for(
                environment.start(force_build=False),
                timeout=build_timeout,
            )

            # 4. Register exec session
            self.exec_sessions[session_token] = environment

            # 5. Send to purple agent
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Agent working..."),
            )
            agent_timeout = task.config.agent.timeout_sec

            if oracle:
                agent_response = await self._run_oracle_locally(
                    task=task,
                    environment=environment,
                    timeout=int(agent_timeout),
                )
            else:
                agent_response = await self._run_agent_session(
                    instruction=task.instruction,
                    environment=environment,
                    agent_url=agent_url,
                    updater=updater,
                    timeout=int(agent_timeout),
                    tag=f"{tag} {task_name}",
                )

            (trial_dir / "agent-transcript.txt").write_text(agent_response or "")

            # 6. Unregister exec session
            self.exec_sessions.pop(session_token, None)

            # 7. Run verification
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag} {task_name}: Verifying..."),
            )
            verifier = Verifier(
                task=task,
                trial_paths=trial_paths,
                environment=environment,
                logger=logger,
            )
            verifier_timeout = task.config.verifier.timeout_sec
            try:
                verifier_result = await asyncio.wait_for(
                    verifier.verify(),
                    timeout=verifier_timeout,
                )
                return verifier_result.rewards or {}
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    _format_verifier_failure(
                        task_name,
                        RuntimeError(f"Verifier timed out after {verifier_timeout}s"),
                        trial_paths,
                    )
                ) from e
            except (
                AddTestsDirError,
                DownloadVerifierDirError,
                RewardFileEmptyError,
                RewardFileNotFoundError,
                VerifierOutputParseError,
            ) as e:
                raise RuntimeError(
                    _format_verifier_failure(task_name, e, trial_paths)
                ) from e

        finally:
            self.exec_sessions.pop(session_token, None)
            if environment:
                try:
                    # delete=False uses plain `docker compose down` (no --rmi all),
                    # avoiding the `GET /images/json` call blocked by the amber docker
                    # gateway policy. Containers and networks are still cleaned up.
                    await environment.stop(delete=False)
                except Exception:
                    pass


    async def _run_oracle_locally(
        self,
        *,
        task: Task,
        environment: BaseEnvironment,
        timeout: int,
    ) -> str:
        solve_path = task.paths.solve_path
        if not solve_path.exists():
            raise FileNotFoundError(f"Solution script not found: {solve_path}")

        await environment.upload_dir(
            source_dir=task.paths.solution_dir,
            target_dir="/solution",
        )
        script = solve_path.read_text()
        result = await environment.exec(
            f"timeout {timeout}s bash /solution/solve.sh",
            timeout_sec=timeout + 15,
        )
        logger.warning(
            "Oracle solve finished: task=%s exit_code=%s stdout_tail=%s stderr_tail=%s",
            task.name,
            result.return_code,
            (result.stdout or "")[-500:],
            (result.stderr or "")[-500:],
        )
        return _format_exec_result(result.return_code, result.stdout or "", result.stderr or "")

    async def _run_agent_session(
        self,
        *,
        instruction: str,
        environment: BaseEnvironment,
        agent_url: str,
        updater: TaskUpdater,
        timeout: int,
        tag: str,
    ) -> str:
        start = time.monotonic()
        transcript: list[str] = []
        outbound = _encode_protocol_message(
            {
                "kind": "task",
                "protocol": "terminal-bench-shell-v1",
                "instruction": instruction,
            }
        )
        new_conversation = True

        while True:
            elapsed = int(time.monotonic() - start)
            remaining = max(1, timeout - elapsed)
            response = await self.messenger.talk_to_agent(
                message=outbound,
                url=agent_url,
                new_conversation=new_conversation,
                timeout=remaining,
            )
            new_conversation = False
            transcript.append(f"green->agent\n{outbound}")
            transcript.append(f"agent->green\n{response}")

            payload = _decode_protocol_message(response)
            kind = payload["kind"]

            if kind == "final":
                return "\n\n".join(transcript)

            if kind != "exec_request":
                raise RuntimeError(f"Unexpected agent payload: {payload}")

            command = payload.get("command")
            if not isinstance(command, str) or not command.strip():
                raise RuntimeError(f"Invalid exec_request: {payload}")

            command_timeout = payload.get("timeout", 30)
            if not isinstance(command_timeout, int):
                raise RuntimeError(f"Invalid timeout in exec_request: {payload}")
            command_timeout = max(1, min(command_timeout, 300))

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"{tag}: $ {command}"),
            )
            result = await environment.exec(command, timeout_sec=command_timeout)
            outbound = _encode_protocol_message(
                {
                    "kind": "exec_result",
                    "exit_code": result.return_code,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                }
            )


def _encode_protocol_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _decode_protocol_message(message: str) -> dict[str, Any]:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return {"kind": "final", "output": message}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object payload, got: {payload!r}")
    return payload


def _format_exec_result(exit_code: int, stdout: str, stderr: str) -> str:
    return (
        f"exit_code={exit_code}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )
