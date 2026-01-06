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

# Option 1: Use environment variable (recommended - keeps key out of shell history)
export GEMINI_API_KEY="your-key-here"
uvx --from git+https://github.com/CJHwong/operating-system-support oss -p gemini "list files"

# Option 2: Pass key directly (visible in shell history and process list)
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

## Security Considerations

### Command Execution

- **All commands require user confirmation** before execution - this is the primary security boundary
- Commands are displayed exactly as they will be executed - review them carefully
- Auto-approval is per-command-base (e.g., approving `ls` does NOT approve `rm`)

### API Key Security

- **Prefer environment variables** over command-line `--api-key` flag
- Command-line arguments are visible in shell history (`~/.bash_history`, `~/.zsh_history`)
- Command-line arguments are visible in process listings (`ps aux`)
- Set `GEMINI_API_KEY` in your shell profile or use a secrets manager

### Limitations

- This tool executes arbitrary shell commands and Python code on your system
- The AI may suggest dangerous commands - always review before approving
- Not suitable for unattended/automated execution without additional safeguards
- Defense-in-depth measures exist but user confirmation is the primary control

## Troubleshooting

### Ollama Issues

**"Ollama connection failed. Is Ollama running?"**

- Ensure Ollama is installed and running: `ollama serve`
- Check if the model exists: `ollama list`
- Pull the model if needed: `ollama pull gpt-oss` (or your chosen model)

**"Unexpected Ollama error"**

- Check Ollama logs for details
- Ensure you have enough memory for the model
- Try a smaller model if resources are constrained

### Gemini Issues

**"Gemini authentication failed"**

- Verify your API key is correct
- Check if the key has been revoked at https://aistudio.google.com/apikey
- Ensure the `GEMINI_API_KEY` environment variable is set correctly

**"Gemini request timed out"**

- Check your internet connection
- The Gemini API may be experiencing issues - try again later
- Consider using Ollama for local operation

## License

MIT License
