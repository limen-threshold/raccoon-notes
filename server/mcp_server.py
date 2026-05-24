"""MCP server for Raccoon Notes — stdio transport.

Exposes one tool:
    generate_lesson(topic, outline, memory_source?) → lesson dict

Plug into Claude Code / claude.ai / SillyTavern / LobeHub / any MCP client.

Setup in claude_desktop_config.json (or equivalent):
    {
      "mcpServers": {
        "raccoon-notes": {
          "command": "python",
          "args": ["-m", "server.mcp_server"],
          "cwd": "/path/to/raccoon-notes",
          "env": {"ANTHROPIC_API_KEY": "sk-ant-..."}
        }
      }
    }

Usage:
    pip install mcp pyyaml
    python -m server.mcp_server
"""
from __future__ import annotations
import os
import sys
import asyncio
import json
from pathlib import Path

# Ensure package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for ANTHROPIC_API_KEY when launched outside an interactive shell
try:
    from dotenv import load_dotenv
    for env_path in (
        Path(__file__).parent.parent / ".env",
        Path.home() / "claude-home" / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    print("[mcp_server] mcp package not installed. Run: pip install mcp", file=sys.stderr)
    raise

from core import pipeline


app = Server("raccoon-notes")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="generate_lesson",
            description=(
                "Generate a personalized lesson for the user, built from their "
                "shared memories with their AI. The lesson explains a topic THROUGH "
                "the user's lived experiences — concepts get illustrated by actual "
                "memories the user has stored, in the register they actually use. "
                "Output includes opening, sections with concept × memory bridges, "
                "and a reflection step that asks the user to look back at the "
                "memories the lesson drew from."
            ),
            inputSchema={
                "type": "object",
                "required": ["topic", "outline"],
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic title (e.g. \"为什么你明明知道自己在拖延还是在刷手机\"). What the user wants to learn.",
                    },
                    "outline": {
                        "type": "array",
                        "description": "Ordered list of concepts that make up this lesson. Each concept is a knowledge point. Use a preset outline if you have one, or generate one for custom topics.",
                        "items": {
                            "type": "object",
                            "required": ["concept"],
                            "properties": {
                                "concept": {
                                    "type": "string",
                                    "description": "Full natural-language description of this knowledge point — what it is, what it claims.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Optional keywords to retrieve memories matching this concept. If omitted, the concept text is used.",
                                },
                            },
                        },
                    },
                    "memory_source": {
                        "type": "object",
                        "description": "Where to retrieve memories from. Defaults to local Anchor at http://localhost:8000.",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["anchor", "mcp_memory", "raw_chat", "none"],
                                "description": "Memory source backend.",
                            },
                            "endpoint": {"type": "string"},
                            "search_path": {"type": "string"},
                            "text": {
                                "type": "string",
                                "description": "Only used when type=raw_chat — the pasted chat transcript.",
                            },
                        },
                    },
                    "n_memories_per_concept": {
                        "type": "integer",
                        "description": "How many memories to retrieve per concept. Default 5.",
                        "default": 5,
                    },
                    "reflection_depth": {
                        "type": "string",
                        "enum": ["deep", "light"],
                        "description": (
                            "How pointed the closing reflection prompts should be. "
                            "'deep' (default) asks excavating questions tied to the cited memory — "
                            "expects the user wants to feel something specific. "
                            "'light' asks gentler, curiosity-shaped questions the user can answer "
                            "briefly or skip — for moments when the user wants the lesson without "
                            "being pried open."
                        ),
                        "default": "deep",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "generate_lesson":
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    if not os.getenv("ANTHROPIC_API_KEY"):
        return [types.TextContent(
            type="text",
            text=json.dumps({"_error": "ANTHROPIC_API_KEY not set in server env"}),
        )]

    topic = arguments.get("topic", "").strip()
    outline_in = arguments.get("outline", [])
    if not topic or not outline_in:
        return [types.TextContent(
            type="text",
            text=json.dumps({"_error": "topic and outline are required"}),
        )]

    outline = [
        {"concept": o.get("concept", ""), "query": o.get("query") or o.get("concept", "")}
        for o in outline_in
        if o.get("concept")
    ]

    # Run the synchronous pipeline in a thread to avoid blocking the event loop
    lesson = await asyncio.to_thread(
        pipeline.run_lesson,
        topic,
        outline,
        arguments.get("memory_source"),
        int(arguments.get("n_memories_per_concept", 5)),
        arguments.get("reflection_depth", "deep"),
    )

    return [types.TextContent(type="text", text=json.dumps(lesson, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="raccoon-notes",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
