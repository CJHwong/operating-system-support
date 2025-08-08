# Operating System Support

OSS uses [gpt-oss](https://openai.com/index/introducing-gpt-oss/) for AI-driven interactions with the operating system.

It can use shell commands and execute Python scripts. Built with [Ollama](https://ollama.ai/) for local AI inference.

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
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py

# One-time query
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py "list files in current directory"
```

### Method 1.5: Create an alias for easier usage

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
alias oss='uvx --from git+https://github.com/CJHwong/operating-system-support oss.py'
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
- [Ollama](https://ollama.ai/) with the `gpt-oss` model installed

### Setting up Ollama

1. Install Ollama from <https://ollama.ai/>
2. Install the required model:

```bash
ollama pull gpt-oss
```

## Command Line Options

```bash
oss.py [OPTIONS] [QUERY]

Options:
  -m, --model MODEL    Model to use (default: gpt-oss)
  -v, --verbose        Enable verbose mode for debugging
  -q, --quiet          Suppress all output except results (query mode only)
  -h, --help           Show help message
```

## Interactive Mode Commands

When running in interactive mode:

- `exit` - Quit the application
- `!show` - Display currently auto-approved commands
- `!clear` - Clear all auto-approved commands

## Examples

```bash
# System information
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py "show me system info"

# File operations
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py "create a backup of important files"

# Python calculations
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py "calculate the fibonacci sequence up to 100"

# Interactive mode
uvx --from git+https://github.com/CJHwong/operating-system-support oss.py
```

## Safety Features

- All commands require user confirmation before execution
- Auto-approval system for repeated safe commands  
- Separate confirmation for Python code execution
- Clear output formatting to show what's being executed

## License

MIT License
