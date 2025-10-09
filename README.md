# Operating System Support

OSS is an AI agent for interacting with your operating system through shell commands and Python execution.

Supports two AI providers:

- [Ollama](https://ollama.ai/) (default, local)
- [Google Gemini](https://ai.google.dev/gemini-api) (cloud)

## Features

- Execute shell commands with user confirmation
- Run Python code interactively
- Auto-approval system for repeated commands
- Built-in query preprocessing for better AI understanding
- Tool-based architecture with JSON schema validation

## Installation & Usage

### Method 1: Direct execution with uvx (Recommended)

Run directly from GitHub without cloning:

```bash
# Interactive mode
uvx --from git+https://github.com/CJHwong/operating-system-support oss

# One-time query
uvx --from git+https://github.com/CJHwong/operating-system-support oss "list files in current directory"
```

### Method 1.5: Create an alias for easier usage

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
alias oss='uvx --from git+https://github.com/CJHwong/operating-system-support oss'
```

Then reload your shell or run `source ~/.zshrc` (or your shell's config file).

Now you can use it simply:

```bash
# Interactive mode
oss

# One-time queries
oss "show system information"
oss "list files and directories"
oss "calculate fibonacci sequence in python"
```

### Method 2: Local installation

1. Clone the repository:

```bash
git clone https://github.com/CJHwong/operating-system-support.git
cd operating-system-support
```

2. Run with uv:

```bash
# Interactive mode
uv run oss.py

# One-time query
uv run oss.py "calculate 2+2 using python"
```

3. Or make it executable and run directly:

```bash
chmod +x oss.py
./oss.py "show system information"
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Python package manager
- An AI provider (choose one):
  - **Ollama** (Local, free): Install from <https://ollama.ai/> and pull a model like `ollama pull gpt-oss`
  - **Google Gemini** (Cloud, free tier available): Get an API key from <https://aistudio.google.com/apikey>

### Quick Setup Examples

#### Using Ollama (Local, Default)

```bash
# Install and start Ollama
ollama pull gpt-oss  # or llama3.2, mistral, etc.

# Run OSS (uses Ollama by default)
uvx --from git+https://github.com/CJHwong/operating-system-support oss "list files"
```

#### Using Google Gemini

```bash
# Get your API key from https://aistudio.google.com/apikey

# Run OSS with Gemini
uvx --from git+https://github.com/CJHwong/operating-system-support oss -p gemini --api-key "your-key-here" "list files"
```

## Command Line Options

```plaintext
oss.py [OPTIONS] [QUERY]

Options:
  -p, --provider PROVIDER   AI provider to use: ollama or gemini (default: ollama)
  -m, --model MODEL         Model to use (default: gpt-oss for Ollama, gemini-flash-latest for Gemini)
  --api-key KEY             API key for Gemini provider (not needed for Ollama)
  -v, --verbose             Enable verbose mode for debugging
  -q, --quiet               Suppress all output except results
  -h, --help                Show help message
```

## Interactive Mode Commands

When running in interactive mode:

- `exit` - Quit the application
- `!show` - Display currently auto-approved commands
- `!clear` - Clear all auto-approved commands

## Examples

```bash
# System information
uvx --from git+https://github.com/CJHwong/operating-system-support oss "show me system info"

# File operations
uvx --from git+https://github.com/CJHwong/operating-system-support oss "create a backup of important files"

# Python calculations
uvx --from git+https://github.com/CJHwong/operating-system-support oss "calculate the fibonacci sequence up to 100"

# Interactive mode
uvx --from git+https://github.com/CJHwong/operating-system-support oss
```

## Safety Features

- All commands require user confirmation before execution
- Auto-approval system for repeated safe commands  
- Separate confirmation for Python code execution
- Clear output formatting to show what's being executed

## License

MIT License
