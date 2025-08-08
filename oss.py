#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ollama>=0.5.3",
# ]
# ///
import argparse
import contextlib
import datetime
import inspect
import io
import json
import locale
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import ollama

DEFAULT_MODEL = 'gpt-oss'


def tool(description: str, param_descriptions: dict[str, str] | None = None) -> Callable:
    """
    Decorator to mark a method as a tool with automatic JSON schema generation.

    Args:
        description: Description of what the tool does
        param_descriptions: Parameter descriptions as a dictionary where keys are parameter names
    """

    if param_descriptions is None:
        param_descriptions = {}

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_type = 'string'  # Default type
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = 'integer'
                elif param.annotation is float:
                    param_type = 'number'
                elif param.annotation is bool:
                    param_type = 'boolean'

            parameters[param_name] = {
                'type': param_type,
                'description': param_descriptions.get(param_name, f'The {param_name} parameter'),
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        setattr(
            func,
            '_tool_definition',
            {
                'type': 'function',
                'function': {
                    'name': func.__name__,
                    'description': description,
                    'parameters': {
                        'type': 'object',
                        'properties': parameters,
                        'required': required,
                    },
                },
            },
        )

        return func

    return decorator


class ToolRegistry:
    """Manages tool registration and discovery."""

    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self._definitions: list[dict[str, Any]] = []

    def register_tools_from_instance(self, instance):
        """Auto-discover and register tools from an instance."""
        self.tools.clear()
        self._definitions.clear()

        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if hasattr(method, '_tool_definition'):
                self.tools[name] = method
                self._definitions.append(getattr(method, '_tool_definition'))

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for Ollama."""
        return self._definitions

    def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Execute a tool by name with given arguments."""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        else:
            return {'error': f'Tool `{tool_name}` not found.'}


class CommandApprover:
    """Handles command confirmation and auto-approval logic."""

    def __init__(self, quiet: bool = False):
        self.auto_approved_commands: set[str] = set()
        self.quiet = quiet

    def get_command_base(self, command: str) -> str:
        """Extract the base command from a full command string."""
        return command.strip().split()[0] if command.strip() else ''

    def confirm_command(self, command: str) -> bool:
        """Show confirmation prompt and handle auto-approval logic."""
        base_command = self.get_command_base(command)

        # Check if this command type is already auto-approved
        if base_command in self.auto_approved_commands:
            return True

        # Show confirmation prompt with terminal notification
        print('\a', end='')  # Bell character to trigger OS notification
        prompt = f"Run? (Enter/y/n/a for auto-approve '{base_command}'): "
        try:
            response = input(prompt).strip().lower()

            if response == '' or response == 'y':
                return True
            elif response == 'a':
                self.auto_approved_commands.add(base_command)
                if not self.quiet:
                    print(f"Auto-approved all '{base_command}' commands for this session")
                return True
            else:
                if not self.quiet:
                    print('Command cancelled')
                return False
        except (EOFError, KeyboardInterrupt):
            if not self.quiet:
                print('\nCommand cancelled')
            return False

    def confirm_python_code(self) -> bool:
        """Handle Python code confirmation (always requires approval)."""
        print('\a', end='')  # Bell character to trigger OS notification
        try:
            response = input('Run? (Enter/y/n): ').strip().lower()
            if response != '' and response != 'y':
                if not self.quiet:
                    print('Command cancelled')
                return False
            return True
        except (EOFError, KeyboardInterrupt):
            if not self.quiet:
                print('\nCommand cancelled')
            return False

    def show_auto_approvals(self):
        """Show current auto-approved commands."""
        if self.auto_approved_commands:
            approved = ', '.join(sorted(self.auto_approved_commands))
            if not self.quiet:
                print(f'Auto-approved commands: {approved}')
        else:
            if not self.quiet:
                print('No auto-approved commands')

    def clear_auto_approvals(self):
        """Clear all auto-approved commands."""
        self.auto_approved_commands.clear()
        if not self.quiet:
            print('Cleared all auto-approvals')


class ConversationManager:
    """Manages conversation messages and Ollama integration."""

    def __init__(self, model: str, fast_model: str | None = None, verbose: bool = False):
        self.model = model
        self.fast_model = fast_model or model
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []

    def initialize_conversation(self, system_message: str):
        """Initialize the conversation with a system message."""
        self.messages = [{'role': 'system', 'content': system_message}]

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({'role': 'user', 'content': content})

    def add_assistant_message(self, content: str, tool_calls: list | None = None):
        """Add an assistant message to the conversation."""
        message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if tool_calls:
            message['tool_calls'] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, result: dict[str, Any]):
        """Add a tool result to the conversation."""
        self.messages.append({'role': 'tool', 'content': json.dumps(result)})

    def chat_with_tools(self, tool_definitions: list[dict[str, Any]]):
        """Send the conversation to Ollama with tool definitions."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=tool_definitions,
            )
            if self.verbose:
                print('Ollama response:', response)
            return response
        except Exception as e:
            raise Exception(f'Error during Ollama chat: {e}')

    def rewrite_query(self, query: str) -> str:
        """Use fast model to rewrite user query as clear instruction."""
        prompt = (
            'Rewrite the following user query as a clear, actionable instruction for an AI agent. '
            "Extract the user's intention and remove ambiguity. "
            'Format your response using the following template, adapting fields as needed:\n\n'
            'Action: <Describe the main action to perform>\n'
            'Details: <Provide any relevant details, context, or parameters>\n'
            'Arguments: <List arguments or options if applicable>\n\n'
            'User query: ' + query
        )
        try:
            response = ollama.chat(
                model=self.fast_model,
                messages=[{'role': 'user', 'content': prompt}],
            )
            rewritten = response['message']['content'].strip()
            if self.verbose:
                print('Original query:', query)
                print('Rewritten query:', rewritten)
            return rewritten
        except Exception as e:
            print(f'Error during preprocessing: {e}')
            return query  # Fallback to original if error


class OSAgent:
    """Main agent class that orchestrates tool execution and conversation."""

    def __init__(self, model: str, fast_model: str | None = None, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet

        # Initialize helper components
        self.tool_registry = ToolRegistry()
        self.command_approver = CommandApprover(quiet=quiet)
        self.conversation_manager = ConversationManager(model, fast_model, verbose)

        # Initialize conversation with system message
        system_message = (
            self._get_env_info() + '\n'
            'You are an AI agent with access to the following tools: run_shell_command and python_interpreter. '
            'Whenever possible, use these tools to answer user queries, especially when a programmatic solution is required. '
            'If a tool is not relevant, you may respond directly.\n'
            'Examples:\n'
            "- If the user asks to list files in a directory, use run_shell_command with 'ls <directory>'.\n"
            '- If the user asks to calculate or process data using Python, use python_interpreter with the appropriate code.\n'
            '- For general questions or when tools are not applicable, respond directly.'
        )
        self.conversation_manager.initialize_conversation(system_message)

        # Register tools from this instance
        self.tool_registry.register_tools_from_instance(self)

    def _get_env_info(self) -> str:
        """Generate environment information string."""
        now = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        home_dir = os.path.expanduser('~')
        cwd = os.getcwd()
        username = os.environ.get('USER', 'unknown')
        lang = locale.getlocale()[0]
        return (
            f'Environment: OS={platform.system()} {platform.release()}, '
            f'Shell={os.environ.get("SHELL", "unknown")}, '
            f'Python={platform.python_version()}, '
            f'DateTime={now}, '
            f'UserHome={home_dir}, '
            f'WorkingDir={cwd}, '
            f'Username={username}, '
            f'Locale={lang}'
        )

    def _print_output(self, message: str, to_stderr: bool = False):
        """Centralized output method."""
        if not self.quiet:
            if to_stderr:
                print(message, file=sys.stderr)
            else:
                print(message)

    def _print_tool_output(self, output: str):
        """Print tool output in a formatted style."""
        if not self.quiet:
            lines = output.strip().split('\n')

            if lines:
                # Print first line with tree connector
                print(f'  ⎿  {lines[0]}')

                # Print remaining lines with proper indentation
                for line in lines[1:6]:  # Show up to first 5 additional lines
                    print(f'     {line}')

                # Show truncation indicator if there are more lines
                if len(lines) > 6:
                    remaining = len(lines) - 6
                    print(f'     … +{remaining} lines')
            else:
                print('  ⎿  (no output)')

            print()  # Add spacing after output

    @tool('Execute a shell command.', {'command': 'The shell command to execute.'})
    def run_shell_command(self, command: str) -> dict:
        """Execute a shell command with user confirmation."""
        print(f'\n>>> Bash({command})')
        if not self.command_approver.confirm_command(command):
            return {'error': 'Command cancelled by user'}

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,  # Don't raise exception on non-zero exit
                capture_output=True,
                text=True,
            )
        except Exception as e:
            error_msg = f'Command execution error: {str(e)}'
            self._print_tool_output(error_msg)
            return {'error': str(e)}

        # Handle output regardless of exit code
        if result.returncode != 0:
            error_output = (
                result.stderr
                if result.stderr
                else f'Command failed with exit code {result.returncode} (no error output)'
            )
            self._print_tool_output(error_output)
            return {'error': f'Exit code {result.returncode}', 'stderr': result.stderr}

        output = result.stdout if result.stdout else 'Command executed successfully'
        self._print_tool_output(output)
        return {'stdout': result.stdout}

    @tool('Execute Python code.', {'code': 'The Python code to execute.'})
    def python_interpreter(self, code: str) -> dict:
        """Execute Python code with user confirmation."""
        print(f'\n>>> Python({repr(code)})')
        if not self.command_approver.confirm_python_code():
            return {'error': 'Command cancelled by user'}

        output_io = io.StringIO()
        with (
            contextlib.redirect_stdout(output_io),
            contextlib.redirect_stderr(output_io),
        ):
            try:
                exec(code, {})
                output = output_io.getvalue()
                if not output:
                    output = 'No output produced.'
            except Exception as e:
                error_output = output_io.getvalue()
                full_error = f'{str(e)}\n{error_output}' if error_output else str(e)
                self._print_tool_output(full_error)
                return {'error': str(e), 'output': error_output}

        self._print_tool_output(output)
        return {'output': output}

    def _process_query(self, query: str) -> tuple[int, str]:
        """Process a user query through the conversation manager and tool registry."""
        self.conversation_manager.add_user_message(query)

        while True:  # Loop to handle multi-step tool use
            try:
                response = self.conversation_manager.chat_with_tools(self.tool_registry.get_tool_definitions())
            except Exception as e:
                return 1, str(e)

            response_message = response['message']

            if not response_message.get('tool_calls'):
                self.conversation_manager.add_assistant_message(response_message['content'])
                return 0, response_message['content']

            # Handle tool calls
            self.conversation_manager.add_assistant_message(
                response_message['content'], response_message.get('tool_calls')
            )

            tool_calls = response_message['tool_calls']
            for tool_call in tool_calls:
                if self.verbose:
                    self._print_output(f'Tool call: {tool_call}')

                tool_name = tool_call['function']['name']
                tool_args = tool_call['function']['arguments']

                tool_output = self.tool_registry.execute_tool(tool_name, **tool_args)
                self.conversation_manager.add_tool_result(tool_output)

                if 'error' in tool_output and tool_name not in self.tool_registry.tools:
                    self._print_output(f'Unknown tool: {tool_name}', to_stderr=True)

    def _execute_single_query(self, user_query: str):
        """Execute a single query with preprocessing and output handling."""
        try:
            processed_query = self.conversation_manager.rewrite_query(user_query)
            status, response = self._process_query(processed_query)

            if status != 0:
                print(response, file=sys.stderr)
            else:
                print(response)

        except Exception as e:
            print(f'Error: {str(e)}', file=sys.stderr)

    def run(self, initial_query: str | None = None):
        """Main execution loop for the agent."""
        if initial_query:
            self._execute_single_query(initial_query)
            return

        self._print_output(
            "Operating System Support - Type 'exit' to quit, '!show' for auto-approvals, '!clear' to clear auto-approvals"
        )

        while True:
            try:
                user_query = input('Enter your query: ')
            except (EOFError, KeyboardInterrupt):
                self._print_output('\nExiting...')
                break

            if user_query.lower() == 'exit':
                break
            elif user_query == '!show':
                self.command_approver.show_auto_approvals()
                continue
            elif user_query == '!clear':
                self.command_approver.clear_auto_approvals()
                continue

            self._execute_single_query(user_query)


def main():
    parser = argparse.ArgumentParser(description='An AI agent that can use tools to interact with the OS.')
    parser.add_argument(
        'query',
        nargs='?',
        default=None,
        help='The query to execute immediately.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose mode for debugging.',
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='Suppress all output except the final result.',
    )
    parser.add_argument(
        '-m',
        '--model',
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL}).',
    )
    args = parser.parse_args()

    # Validate quiet flag usage
    if args.quiet and args.query is None:
        parser.error('--quiet/-q can only be used with a query argument. Use interactive mode without --quiet.')

    agent = OSAgent(
        model=args.model,
        verbose=args.verbose,
        quiet=args.quiet,
    )
    agent.run(initial_query=args.query)


if __name__ == '__main__':
    main()
