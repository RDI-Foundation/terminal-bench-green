from typing import Any
from pathlib import Path
import re
import sys
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import (
    _format_verifier_failure,
    _get_shard_task_names,
    _get_task_names,
    _missing_task_config_message,
    _normalize_config,
    _read_text_tail,
)
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import RewardFileNotFoundError


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


def test_normalize_config_accepts_nested_assessment_config():
    config = {
        "assessment_config": {
            "tasks": ["fix-git", "fix-make"],
            "exclude": ["fix-make"],
            "oracle": True,
            "num_shards": 2,
            "shard_index": 0,
        }
    }

    normalized = _normalize_config(config)

    assert normalized == config["assessment_config"]
    assert _get_task_names(normalized) == ["fix-git"]


def test_get_shard_task_names_round_robin():
    task_names = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]

    assert _get_shard_task_names(task_names, {"num_shards": 3, "shard_index": 0}) == [
        "t0",
        "t3",
        "t6",
    ]
    assert _get_shard_task_names(task_names, {"num_shards": 3, "shard_index": 1}) == [
        "t1",
        "t4",
    ]
    assert _get_shard_task_names(task_names, {"num_shards": 3, "shard_index": 2}) == [
        "t2",
        "t5",
    ]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"num_shards": 0, "shard_index": 0}, "'num_shards' must be >= 1"),
        ({"num_shards": 2, "shard_index": -1}, "'shard_index' must be in [0, num_shards)"),
        ({"num_shards": 2, "shard_index": 2}, "'shard_index' must be in [0, num_shards)"),
    ],
)
def test_get_shard_task_names_rejects_invalid_config(config, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        _get_shard_task_names(["t0", "t1"], config)


def test_missing_task_config_message_mentions_received_keys():
    message = _missing_task_config_message({"assessment_config": {}, "oracle": False})

    assert "assessment_config" in message
    assert "oracle" in message


def test_read_text_tail_returns_suffix_for_long_files(tmp_path):
    log_path = tmp_path / "test-stdout.txt"
    log_path.write_text("a" * 5000 + "tail")

    tail = _read_text_tail(log_path, max_chars=8)

    assert tail == "...\naaaatail"


def test_format_verifier_failure_includes_test_output(tmp_path):
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    trial_paths.test_stdout_path.write_text("line 1\nline 2\nreward missing")

    message = _format_verifier_failure(
        "build-pmars",
        RewardFileNotFoundError("No reward file found"),
        trial_paths,
    )

    assert "Verifier failed for build-pmars" in message
    assert "No reward file found" in message
    assert "test.sh output:" in message
    assert "reward missing" in message


def test_format_verifier_failure_omits_empty_test_output(tmp_path):
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    trial_paths.test_stdout_path.write_text("")

    message = _format_verifier_failure(
        "make-doom-for-mips",
        RuntimeError("Verifier timed out after 900s"),
        trial_paths,
    )

    assert message == (
        "Verifier failed for make-doom-for-mips: "
        "Verifier timed out after 900s"
    )
