FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://get.docker.com | sh && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -f -g 999 docker && adduser agent && usermod -aG docker agent && \
    chown root:docker /var/run && chmod 775 /var/run
RUN mkdir -p /workspace && chown agent:agent /workspace
USER agent
WORKDIR /home/agent

# Pre-bake the task repo so green doesn't need outbound git access at runtime.
RUN git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /home/agent/terminal-bench-2

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
