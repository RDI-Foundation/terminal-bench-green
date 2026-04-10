"""DockerEnvironment subclass for DOOD (Docker-outside-of-Docker).

When green runs inside an amber container and talks to the host Docker daemon
via a mounted socket, bind-mount paths in Harbor's default compose files point
to paths that exist only inside green's container — the host daemon cannot see
them.  This subclass swaps the base compose file for one without bind mounts
and overrides ``is_mounted`` so Harbor uses ``docker compose cp`` to transfer
files in and out of task containers.
"""

import asyncio
from pathlib import Path

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.docker import COMPOSE_BUILD_PATH, COMPOSE_PREBUILT_PATH, COMPOSE_NO_NETWORK_PATH
from harbor.models.trial.paths import EnvironmentPaths


_DOOD_COMPOSE_BASE = Path(__file__).with_name("dood_compose_base.yaml")


class DoodDockerEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = _DOOD_COMPOSE_BASE
    _DOCKER_COMPOSE_BUILD_PATH = COMPOSE_BUILD_PATH
    _DOCKER_COMPOSE_PREBUILT_PATH = COMPOSE_PREBUILT_PATH
    _DOCKER_COMPOSE_NO_NETWORK_PATH = COMPOSE_NO_NETWORK_PATH

    @property
    def is_mounted(self) -> bool:
        return False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        env = {**(env or {}), "DEBIAN_FRONTEND": "noninteractive"}
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def start(self, force_build: bool):
        self._use_prebuilt = not force_build and self.task_env_config.docker_image

        if not self._use_prebuilt:
            lock = self._image_build_locks.setdefault(
                self.environment_name, asyncio.Lock()
            )
            async with lock:
                await self._run_docker_compose_command(["build"])

        # Clean up only this project's containers.
        try:
            await self._run_docker_compose_command(["down", "--remove-orphans"])
        except RuntimeError:
            pass

        await self._run_docker_compose_command(["up", "--detach", "--wait"])

        # Create the log directories that bind mounts would have provided.
        await self.exec(
            f"mkdir -p {EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.agent_dir} "
            f"{EnvironmentPaths.artifacts_dir}"
        )
