"""Gradio Web UI for Orchestrator.

Provides a web interface for interacting with the orchestrator:
- Chat with streaming responses
- Editable code artifacts
- Routing visualization
- Remote access via gradio.live
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Generator

import gradio as gr
import httpx

# Default API base URL — sourced from centralized config
from src.config import get_config as _get_config

DEFAULT_API_URL = _get_config().server_urls.api_url


def create_httpx_client(api_url: str) -> httpx.Client:
    """Create an HTTP client for API calls."""
    return httpx.Client(base_url=api_url, timeout=300.0)


def stream_chat_response(
    message: str,
    history: list[list[str]],
    api_url: str,
) -> Generator[tuple[list[list[str]], str | None, dict | None], None, None]:
    """Stream chat responses from the orchestrator API.

    Yields:
        Tuple of (history, code_artifact, routing_metadata)
    """
    if not message.strip():
        yield history, None, None
        return

    # Add user message to history
    history = history + [[message, ""]]

    # Collected state
    code_artifact = None
    routing_metadata: dict[str, Any] = {
        "turns": [],
        "total_tokens": 0,
        "total_elapsed_ms": 0,
    }
    current_turn: dict[str, Any] = {}

    try:
        with httpx.Client(base_url=api_url, timeout=300.0) as client:
            with client.stream(
                "POST",
                "/chat/stream",
                json={"prompt": message},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")

                    if event_type == "turn_start":
                        current_turn = {
                            "turn": data.get("turn"),
                            "role": data.get("role"),
                            "tokens": 0,
                            "elapsed_ms": 0,
                        }

                    elif event_type == "token":
                        content = data.get("content", "")
                        history[-1][1] += content
                        yield history, code_artifact, routing_metadata

                    elif event_type == "tool":
                        # Tool call - add to history
                        tool_name = data.get("name", "unknown")
                        tool_result = data.get("result", "")
                        history[-1][1] += f"\n[Tool: {tool_name}]\n{tool_result}\n"
                        yield history, code_artifact, routing_metadata

                    elif event_type == "file":
                        # File artifact - extract code
                        content = data.get("content", "")
                        path = data.get("path", "")
                        action = data.get("action", "create")
                        code_artifact = f"# {action.upper()}: {path}\n{content}"
                        yield history, code_artifact, routing_metadata

                    elif event_type == "turn_end":
                        current_turn["tokens"] = data.get("tokens", 0)
                        current_turn["elapsed_ms"] = data.get("elapsed_ms", 0)
                        routing_metadata["turns"].append(current_turn)
                        routing_metadata["total_tokens"] += current_turn["tokens"]
                        routing_metadata["total_elapsed_ms"] += current_turn["elapsed_ms"]
                        yield history, code_artifact, routing_metadata

                    elif event_type == "final":
                        answer = data.get("answer", "")
                        if answer and answer != history[-1][1]:
                            history[-1][1] = answer
                        yield history, code_artifact, routing_metadata

                    elif event_type == "error":
                        error_msg = data.get("message", "Unknown error")
                        history[-1][1] += f"\n\n**Error:** {error_msg}"
                        yield history, code_artifact, routing_metadata

    except httpx.HTTPStatusError as e:
        history[-1][1] = f"API Error: {e.response.status_code} - {e.response.text}"
        yield history, None, None
    except httpx.RequestError as e:
        history[-1][1] = f"Connection Error: {e}"
        yield history, None, None


def sync_chat(
    message: str,
    history: list[list[str]],
    api_url: str,
) -> tuple[list[list[str]], str | None, dict | None]:
    """Non-streaming chat for fallback."""
    if not message.strip():
        return history, None, None

    history = history + [[message, ""]]

    try:
        with httpx.Client(base_url=api_url, timeout=300.0) as client:
            response = client.post(
                "/chat",
                json={"prompt": message, "mock_mode": True},
            )
            response.raise_for_status()
            data = response.json()
            history[-1][1] = data.get("answer", "No response")
            return history, None, {"turns": data.get("turns", 1)}
    except Exception as e:
        history[-1][1] = f"Error: {e}"
        return history, None, None


def create_ui(api_url: str = DEFAULT_API_URL) -> gr.Blocks:
    """Create the Gradio Blocks interface."""

    with gr.Blocks(
        title="Orchestrator",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css="""
        .code-editor textarea {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 14px;
        }
        .routing-panel {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        }
        """,
    ) as demo:
        # Header
        gr.Markdown(
            """
            # Orchestrator
            Hierarchical LLM orchestration with role-based routing.
            """
        )

        # State
        api_url_state = gr.State(value=api_url)

        with gr.Row():
            # Left column: Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask me to code something...",
                        lines=3,
                        scale=4,
                        show_label=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear", size="sm")
                    undo_btn = gr.Button("Undo", size="sm")

            # Right column: Artifacts + Routing
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Code"):
                        gr.Markdown("### Generated Code")
                        code_display = gr.Code(
                            label="",
                            language="python",
                            interactive=True,
                            lines=20,
                            elem_classes=["code-editor"],
                        )
                        with gr.Row():
                            save_btn = gr.Button("Save", size="sm")
                            copy_code_btn = gr.Button("Copy", size="sm")

                    with gr.Tab("Routing"):
                        gr.Markdown("### Orchestration Flow")
                        routing_display = gr.JSON(
                            label="",
                            elem_classes=["routing-panel"],
                        )

                    with gr.Tab("Settings"):
                        gr.Markdown("### API Settings")
                        api_url_input = gr.Textbox(
                            label="API URL",
                            value=api_url,
                            info="Base URL for the orchestrator API",
                        )
                        update_url_btn = gr.Button("Update URL", size="sm")

                        gr.Markdown("### Status")
                        health_status = gr.Markdown("Checking...")
                        check_health_btn = gr.Button("Check Health", size="sm")

        # Event handlers
        def submit_message(message: str, history: list[list[str]], url: str):
            """Handle message submission with streaming."""
            if not message.strip():
                return history, None, None, ""

            # Return generator results
            for result in stream_chat_response(message, history, url):
                yield result + ("",)  # Add empty message box

        def clear_chat():
            """Clear chat history."""
            return [], None, None

        def undo_last():
            """Remove last exchange."""
            return gr.update(value=lambda h: h[:-1] if h else [])

        def update_api_url(new_url: str):
            """Update the API URL."""
            return new_url, f"API URL updated to: {new_url}"

        def check_health(url: str):
            """Check API health status."""
            try:
                with httpx.Client(base_url=url, timeout=10.0) as client:
                    response = client.get("/health")
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get("status", "unknown")
                        models = data.get("models_loaded", 0)
                        return f"**Status:** {status}\n**Models Loaded:** {models}"
                    else:
                        return f"**Error:** HTTP {response.status_code}"
            except Exception as e:
                return f"**Connection Failed:** {e}"

        def save_code(code: str):
            """Save code to file (placeholder)."""
            if not code:
                return gr.Warning("No code to save")
            # TODO: Implement file save dialog
            return gr.Info(f"Code saved ({len(code)} chars)")

        def copy_code(code: str):
            """Copy code to clipboard (handled client-side)."""
            if not code:
                return gr.Warning("No code to copy")
            return gr.Info("Code copied to clipboard")

        # Wire up events
        submit_btn.click(
            fn=submit_message,
            inputs=[msg, chatbot, api_url_state],
            outputs=[chatbot, code_display, routing_display, msg],
        )

        msg.submit(
            fn=submit_message,
            inputs=[msg, chatbot, api_url_state],
            outputs=[chatbot, code_display, routing_display, msg],
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, code_display, routing_display],
        )

        update_url_btn.click(
            fn=update_api_url,
            inputs=[api_url_input],
            outputs=[api_url_state, health_status],
        )

        check_health_btn.click(
            fn=check_health,
            inputs=[api_url_state],
            outputs=[health_status],
        )

        save_btn.click(
            fn=save_code,
            inputs=[code_display],
        )

        copy_code_btn.click(
            fn=copy_code,
            inputs=[code_display],
        )

        # Check health on load
        demo.load(
            fn=check_health,
            inputs=[api_url_state],
            outputs=[health_status],
        )

    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Orchestrator Gradio UI")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Base URL for the orchestrator API",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public gradio.live URL",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help="Username:password for basic auth (e.g., 'admin:secret')",
    )

    args = parser.parse_args()

    # Parse auth if provided
    auth = None
    if args.auth:
        parts = args.auth.split(":", 1)
        if len(parts) == 2:
            auth = (parts[0], parts[1])
        else:
            print(f"Warning: Invalid auth format '{args.auth}', expected 'user:pass'")

    # Create and launch UI
    demo = create_ui(api_url=args.api_url)

    print(f"\n{'=' * 60}")
    print("ORCHESTRATOR WEB UI")
    print(f"{'=' * 60}")
    print(f"API URL: {args.api_url}")
    print(f"Local URL: http://{args.host}:{args.port}")
    if args.share:
        print("Public URL: Creating gradio.live link...")
    if auth:
        print(f"Auth: {auth[0]}:{'*' * len(auth[1])}")
    print(f"{'=' * 60}\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
    )


if __name__ == "__main__":
    main()
