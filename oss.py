#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ollama>=0.5.3",
#   "google-genai>=1.42.0",
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
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ollama
from google import genai
from google.genai import types

DEFAULT_MODEL = 'gpt-oss'


# ============================================================================
# Response Data Classes
# ============================================================================


@dataclass
class ToolCall:
    """Represents a tool/function call request from the AI."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Standardized response format from AI providers."""

    content: str = ''
    tool_calls: list[ToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def has_content(self) -> bool:
        """Check if response contains text content."""
        return self.content != ''


# ============================================================================
# Provider Enum
# ============================================================================


class Provider(Enum):
    """Supported AI providers."""

    OLLAMA = 'ollama'
    GEMINI = 'gemini'


# ============================================================================
# Tool Decorator
# ============================================================================


def tool(
    description: str, param_descriptions: dict[str, str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to mark a method as a tool with automatic JSON schema generation.

    Args:
        description: Description of what the tool does
        param_descriptions: Parameter descriptions as a dictionary where keys are parameter names
    """

    if param_descriptions is None:
        param_descriptions = {}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
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

        # Build parameters schema - Gemini-friendly format
        param_schema: dict[str, Any] = {
            'type': 'object',
            'properties': parameters,
            'additionalProperties': False,  # Strict schema for Gemini compatibility
        }
        if required:  # Only add 'required' field if there are required parameters
            param_schema['required'] = required

        setattr(
            func,
            '_tool_definition',
            {
                'type': 'function',
                'function': {
                    'name': func.__name__,
                    'description': description,
                    'parameters': param_schema,
                },
            },
        )

        return func

    return decorator


# ============================================================================
# AI Provider Clients
# ============================================================================


class OllamaClient:
    """Client for Ollama using native ollama SDK."""

    def __init__(self, model: str, verbose: bool = False):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., 'llama3.2', 'gpt-oss')
            verbose: Enable verbose logging
        """
        self.model = model
        self.verbose = verbose

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        """
        Send chat request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional tool definitions for function calling

        Returns:
            ChatResponse object
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            if self.verbose:
                print('Ollama response:', response)

            # Convert to ChatResponse
            message = response['message']
            content = message.get('content', '')

            tool_calls = None
            if message.get('tool_calls'):
                tool_calls = [
                    ToolCall(
                        id=tc.get('id', ''),
                        name=tc['function']['name'],
                        arguments=json.loads(tc['function']['arguments'])
                        if isinstance(tc['function']['arguments'], str)
                        else tc['function']['arguments'],
                    )
                    for tc in message['tool_calls']
                ]

            return ChatResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            raise Exception(f'Ollama API error: {str(e)}')


class GeminiClient:
    """Client for Google Gemini using native google-genai SDK (v1.42+)."""

    def __init__(self, model: str, api_key: str, verbose: bool = False):
        """
        Initialize Gemini client.

        Args:
            model: Model name (e.g., 'gemini-2.5-flash')
            api_key: Google AI API key
            verbose: Enable verbose logging
        """
        self.model_name = model
        self.verbose = verbose
        self.client = genai.Client(api_key=api_key)
        self.system_instruction = None

    def _convert_tools_to_gemini_format(self, tools: list[dict[str, Any]] | None) -> list[types.Tool] | None:
        """Convert tool definitions to Gemini format."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool['type'] == 'function':
                func_def = tool['function']
                # Convert parameters dict to Schema object
                params_dict = func_def['parameters']
                schema = self._convert_dict_to_schema(params_dict)
                # Create FunctionDeclaration using parameters with Schema type
                func_decl = types.FunctionDeclaration(
                    name=func_def['name'],
                    description=func_def['description'],
                    parameters=schema,
                )
                function_declarations.append(func_decl)

        return [types.Tool(function_declarations=function_declarations)] if function_declarations else None

    def _convert_dict_to_schema(self, schema_dict: dict[str, Any]) -> types.Schema:
        """Convert a JSON schema dictionary to Gemini Schema type."""
        # Map JSON schema types to Gemini Schema types
        type_mapping = {
            'object': types.Type.OBJECT,
            'string': types.Type.STRING,
            'integer': types.Type.INTEGER,
            'number': types.Type.NUMBER,
            'boolean': types.Type.BOOLEAN,
            'array': types.Type.ARRAY,
        }

        schema_type = type_mapping.get(schema_dict.get('type', 'object').lower(), types.Type.OBJECT)

        # Build properties if they exist
        properties = {}
        if 'properties' in schema_dict:
            for prop_name, prop_schema in schema_dict['properties'].items():
                properties[prop_name] = self._convert_dict_to_schema(prop_schema)

        # Build the schema
        return types.Schema(
            type=schema_type,
            description=schema_dict.get('description'),
            properties=properties if properties else None,
            required=schema_dict.get('required'),
            items=self._convert_dict_to_schema(schema_dict['items']) if 'items' in schema_dict else None,
        )

    def _convert_messages_to_contents(self, messages: list[dict[str, Any]]) -> list[types.Content]:
        """Convert message history to Gemini Content format."""
        contents = []

        for msg in messages:
            role = msg['role']

            if role == 'system':
                # Store system instruction separately
                self.system_instruction = msg['content']
            elif role == 'user':
                contents.append(types.Content(role='user', parts=[types.Part(text=msg['content'])]))
            elif role == 'assistant':
                # Assistant messages with optional tool calls
                parts = []
                if msg.get('content'):
                    parts.append(types.Part(text=msg['content']))

                if msg.get('tool_calls'):
                    for tool_call in msg['tool_calls']:
                        args = tool_call['function']['arguments']
                        if isinstance(args, str):
                            args = json.loads(args)
                        parts.append(types.Part.from_function_call(name=tool_call['function']['name'], args=args))

                if parts:
                    contents.append(types.Content(role='model', parts=parts))
            elif role == 'tool':
                # Tool result
                result_data = json.loads(msg['content']) if isinstance(msg['content'], str) else msg['content']
                contents.append(
                    types.Content(
                        role='user',
                        parts=[
                            types.Part.from_function_response(name=msg.get('name', 'unknown'), response=result_data)
                        ],
                    )
                )

        return contents

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        """
        Send chat request to Gemini.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions

        Returns:
            ChatResponse object
        """
        try:
            # Convert messages and tools
            contents = self._convert_messages_to_contents(messages)
            gemini_tools = self._convert_tools_to_gemini_format(tools)

            # Build config - be explicit to avoid mypy type issues
            config: types.GenerateContentConfig | None = None
            if self.system_instruction and gemini_tools:
                config = types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=gemini_tools,  # type: ignore[arg-type]
                )
            elif self.system_instruction:
                config = types.GenerateContentConfig(system_instruction=self.system_instruction)
            elif gemini_tools:
                config = types.GenerateContentConfig(tools=gemini_tools)  # type: ignore[arg-type]

            # Generate content
            response = self.client.models.generate_content(model=self.model_name, contents=contents, config=config)  # type: ignore[arg-type]

            if self.verbose:
                print('Gemini response:', response)

            # Extract text content and function calls manually from parts
            # This avoids the warning when accessing response.text with function calls present
            content = ''
            tool_calls = None

            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                text_parts = []
                extracted_calls = []

                for i, part in enumerate(response.candidates[0].content.parts):
                    if part.text:
                        text_parts.append(part.text)
                    elif part.function_call:
                        fc = part.function_call
                        # Add null checks for name and args
                        if fc.name and fc.args is not None:
                            extracted_calls.append(ToolCall(id=f'call_{i}', name=fc.name, arguments=dict(fc.args)))

                # Combine text parts if any
                if text_parts:
                    content = ''.join(text_parts)

                # Set tool_calls if any
                if extracted_calls:
                    tool_calls = extracted_calls

            return ChatResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            raise Exception(f'Gemini API error: {str(e)}')


# ============================================================================
# Tool Management
# ============================================================================


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
        """Get all tool definitions for the API."""
        return self._definitions

    def execute_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Execute a tool by name with given arguments."""
        if tool_name in self.tools:
            return self.tools[tool_name](**kwargs)
        else:
            return {'error': f'Tool `{tool_name}` not found.'}


# ============================================================================
# Command Approval
# ============================================================================


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


# ============================================================================
# Conversation Management
# ============================================================================


class ConversationManager:
    """Manages conversation messages and AI provider integration."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        fast_model: str | None = None,
        verbose: bool = False,
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.fast_model = fast_model or model
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []

        # Create appropriate client based on provider
        self.client: OllamaClient | GeminiClient
        if provider == Provider.OLLAMA:
            self.client = OllamaClient(model=model, verbose=verbose)
        elif provider == Provider.GEMINI:
            if not api_key:
                raise ValueError('API key required for Gemini provider')
            self.client = GeminiClient(model=model, api_key=api_key, verbose=verbose)
        else:
            raise ValueError(f'Unsupported provider: {provider}')

    def initialize_conversation(self, system_message: str):
        """Initialize the conversation with a system message."""
        self.messages = [{'role': 'system', 'content': system_message}]

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({'role': 'user', 'content': content})

    def add_assistant_message(self, content: str, tool_calls: list | None = None):
        """Add an assistant message to the conversation."""
        message: dict[str, Any] = {'role': 'assistant'}
        # Only add content if it's non-empty
        if content:
            message['content'] = content
        if tool_calls:
            message['tool_calls'] = tool_calls
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: dict[str, Any]):
        """Add a tool result to the conversation."""
        self.messages.append(
            {
                'role': 'tool',
                'tool_call_id': tool_call_id,
                'name': tool_name,  # Gemini needs the function name
                'content': json.dumps(result),
            }
        )

    def chat_with_tools(self, tool_definitions: list[dict[str, Any]]):
        """Send the conversation to the API with tool definitions."""
        try:
            response = self.client.chat(
                messages=self.messages,
                tools=tool_definitions,
            )
            if self.verbose:
                print('API response:', response)
            return response
        except Exception as e:
            raise Exception(f'Error during API chat: {e}')

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
            response = self.client.chat(
                messages=[{'role': 'user', 'content': prompt}],
            )
            rewritten = response.content.strip() if response.has_content else query
            if self.verbose:
                print('Original query:', query)
                print('Rewritten query:', rewritten)
            return rewritten
        except Exception as e:
            print(f'Error during preprocessing: {e}')
            return query  # Fallback to original if error


# ============================================================================
# Main Agent
# ============================================================================


class OSAgent:
    """Main agent class that orchestrates tool execution and conversation."""

    def __init__(
        self,
        provider: Provider,
        model: str,
        fast_model: str | None = None,
        verbose: bool = False,
        quiet: bool = False,
        api_key: str | None = None,
    ):
        self.verbose = verbose
        self.quiet = quiet

        # Initialize helper components
        self.tool_registry = ToolRegistry()
        self.command_approver = CommandApprover(quiet=quiet)
        self.conversation_manager = ConversationManager(
            provider=provider,
            model=model,
            fast_model=fast_model,
            verbose=verbose,
            api_key=api_key,
        )

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

            # Check if we have tool calls
            if not response.has_tool_calls:
                # No tool calls - return the content
                if not response.has_content:
                    error_msg = 'API returned empty response'
                    return 1, error_msg
                self.conversation_manager.add_assistant_message(response.content)
                return 0, response.content

            # Handle tool calls
            # Convert tool calls back to dict format for add_assistant_message
            # Keep arguments as dict for Ollama, Gemini converts in _convert_messages_to_contents
            tool_calls = response.tool_calls or []
            tool_calls_dict = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': tc.arguments,  # Keep as dict, don't stringify
                    },
                }
                for tc in tool_calls
            ]

            self.conversation_manager.add_assistant_message(response.content, tool_calls_dict)

            for tool_call in tool_calls:
                if self.verbose:
                    self._print_output(f'Tool call: {tool_call.name}({tool_call.arguments})')

                tool_output = self.tool_registry.execute_tool(tool_call.name, **tool_call.arguments)
                self.conversation_manager.add_tool_result(tool_call.id, tool_call.name, tool_output)

                if 'error' in tool_output and tool_call.name not in self.tool_registry.tools:
                    self._print_output(f'Unknown tool: {tool_call.name}', to_stderr=True)

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


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='An AI agent that can use tools to interact with the OS using Ollama or Gemini.'
    )
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
        '-p',
        '--provider',
        choices=['ollama', 'gemini'],
        default='ollama',
        help='AI provider to use (default: ollama).',
    )
    parser.add_argument(
        '-m',
        '--model',
        default=DEFAULT_MODEL,
        help=f'Model to use (default: {DEFAULT_MODEL} for Ollama, gemini-flash-latest for Gemini).',
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='API key for Gemini provider. Not needed for Ollama.',
    )
    args = parser.parse_args()

    # Validate quiet flag usage
    if args.quiet and args.query is None:
        parser.error('--quiet/-q can only be used with a query argument. Use interactive mode without --quiet.')

    # Convert provider string to enum
    provider = Provider.OLLAMA if args.provider == 'ollama' else Provider.GEMINI

    # Set default model based on provider if not specified
    model = args.model
    fast_model = model
    if model == DEFAULT_MODEL and provider == Provider.GEMINI:
        model = 'gemini-flash-latest'
        fast_model = 'gemini-flash-lite-latest'

    # Validate Gemini API key
    if provider == Provider.GEMINI and not args.api_key:
        parser.error('Gemini provider requires --api-key environment variable.')

    agent = OSAgent(
        provider=provider,
        model=model,
        fast_model=fast_model,
        verbose=args.verbose,
        quiet=args.quiet,
        api_key=args.api_key,
    )
    agent.run(initial_query=args.query)


if __name__ == '__main__':
    main()
