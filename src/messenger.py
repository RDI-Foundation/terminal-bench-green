import brotli
import gzip
import json
import zlib
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
    DataPart,
)


DEFAULT_TIMEOUT = 300


def _try_decompress(data: bytes, encoding: str) -> bytes:
    """Attempt to decompress data according to Content-Encoding, falling back to raw bytes."""
    encoding = encoding.strip().lower()
    try:
        if encoding == "gzip":
            return gzip.decompress(data)
        if encoding == "deflate":
            return zlib.decompress(data)
        if encoding == "br":
            return brotli.decompress(data)
    except Exception:
        pass
    return data


class _StripContentEncodingTransport(httpx.AsyncHTTPTransport):
    """Strip Content-Encoding and decompress if needed.

    The amber router may rewrite response bodies while leaving a stale
    Content-Encoding header. httpx then either fails to decompress (body
    already plain) or skips decompression (if we just strip the header but
    the body is actually compressed). This transport reads the raw body,
    decompresses if necessary, and returns plain bytes with no
    Content-Encoding header so httpx uses IdentityDecoder.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle_async_request(request)
        encoding = response.headers.get("content-encoding")
        if encoding:
            raw = await response.aread()
            body = _try_decompress(raw, encoding)
            headers = [
                (k, v)
                for k, v in response.headers.raw
                if k.lower() != b"content-encoding"
            ]
            return httpx.Response(
                status_code=response.status_code,
                headers=headers,
                content=body,
                extensions=response.extensions,
            )
        return response


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    consumer: Consumer | None = None,
):
    """Returns dict with context_id, response and status (if exists)"""
    async with httpx.AsyncClient(
        timeout=timeout,
        transport=_StripContentEncodingTransport(),
    ) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        agent_card.url = base_url  # override to ensure A2A messages route via proxy
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {"response": "", "context_id": None}

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


class Messenger:
    def __init__(self):
        self._context_ids = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Communicate with another agent by sending a message and receiving their response.

        Args:
            message: The message to send to the agent
            url: The agent's URL endpoint
            new_conversation: If True, start fresh conversation; if False, continue existing conversation
            timeout: Timeout in seconds for the request (default: 300)

        Returns:
            str: The agent's response message
        """
        outputs = await send_message(
            message=message,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url, None),
            timeout=timeout,
            streaming=True,
        )
        if outputs.get("status", "completed") != "completed":
            raise RuntimeError(f"{url} responded with: {outputs}")
        self._context_ids[url] = outputs.get("context_id", None)
        return outputs["response"]

    def reset(self):
        self._context_ids = {}
