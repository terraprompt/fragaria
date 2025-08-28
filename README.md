## Fragaria - From 'r's in Strawberry to Complex Problem-Solving AI
**Advanced Chain of Thought (CoT) Reasoning API with Reinforcement Learning (RL)**

![Fragaria Logo](header.png)

Fragaria is a powerful and flexible Chain of Thought (CoT) reasoning library that leverages various Language Model (LLM) providers and incorporates Reinforcement Learning (RL) techniques to solve complex problems and answer intricate questions. Named after the botanical genus of strawberries, Fragaria pays homage to the famous "How many 'r's in strawberry?" problem, symbolizing its ability to tackle both simple and complex queries with equal finesse.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [As a Library](#as-a-library)
  - [Command Line Interface](#command-line-interface)
  - [Web Service](#web-service)
- [API Documentation](#api-documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Groq, and Together.ai as LLM providers.
- **Chain of Thought Reasoning**: Employ sophisticated CoT techniques to break down and solve complex problems.
- **Reinforcement Learning**: Utilize RL algorithms to continuously improve problem-solving strategies and adapt to new challenges.
- **Adaptive Learning**: Enhance performance over time through a SQLite-based scoring system integrated with RL techniques.
- **Configurable**: Easy-to-update YAML configuration file for flexible setup of both LLM and RL parameters.
- **OpenAPI Documentation**: Comprehensive API documentation with Swagger UI and ReDoc.
- **CORS Support**: Built-in Cross-Origin Resource Sharing for easy integration with web applications.
- **CLI Tools**: Command-line interface for easy testing and integration.
- **Python Library**: Usable as a Python library in your own projects.

## Installation

Install Fragaria using pip:

```bash
pip install fragaria
```

Or if you want to install from source:

```bash
git clone https://github.com/terraprompt/fragaria.git
cd fragaria
poetry install
```

## Configuration

1. Create a configuration file by copying the default:
   ```bash
   # If installed via pip
   cp /path/to/site-packages/fragaria/config.yaml ./config.yaml
   
   # If installed from source
   cp fragaria/fragaria/config.yaml ./config.yaml
   ```

2. Open `config.yaml` and update the following settings:
   - Set your preferred `llm_provider` (openai, groq, or together)
   - Add your API keys for the respective providers
   - Adjust the model names if necessary
   - Modify the database path and server settings if needed

**Important**: You must update the configuration file with your actual API keys for the LLM provider you want to use. The default values are placeholders and will not work.

## Usage

### As a Library

Fragaria can be used as a Python library in your own projects:

```python
import asyncio
from fragaria import analyze_problem

async def main():
    result = await analyze_problem("How many 'r's in strawberry?")
    print(result["result"])

asyncio.run(main())
```

For more examples, see the `example.py` file in the repository.

You can also use the `FragariaCore` class for more advanced usage:

```python
import asyncio
from fragaria import FragariaCore

async def main():
    # Initialize with a custom config file path (optional)
    core = FragariaCore("path/to/your/config.yaml")
    result = await core.parallel_cot_reasoning("How many 'r's in strawberry?")
    print(result["result"])

asyncio.run(main())
```

**Note**: Before running the examples, you must configure your API keys in the `config.yaml` file. See the [Configuration](#configuration) section for details.

### Command Line Interface

After installation, you can use the `fragaria` command to analyze problems:

```bash
# Analyze a simple problem
fragaria "How many 'r's in strawberry?"

# Use with a system prompt
fragaria "What is the capital of France?" --system-prompt "You are a helpful geography assistant."

# Read from stdin
echo "A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess?" | fragaria

# Get JSON output
fragaria "How many 'r's in strawberry?" --output-format json
```

### Web Service

Start the Fragaria API server:

```bash
# Using the CLI command
fragaria-server

# Or directly with Python
python -m fragaria.main
```

The API will be available at `http://localhost:8000` (or the host/port specified in your config).

You can now send POST requests to `http://localhost:8000/v1/chat/completions` to use the Chain of Thought reasoning capabilities.

## API Documentation

Fragaria provides comprehensive API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON schema: `http://localhost:8000/openapi.json`

## Examples

Here are some sample problems you can solve using Fragaria:

1. The classic strawberry problem:
   ```json
   {
     "model": "faragia-dev",
     "messages": [
       {"role": "user", "content": "How many 'r's in strawberry?"}
     ]
   }
   ```

2. A more complex age-related puzzle:
   ```json
   {
     "model": "faragia-dev",
     "messages": [
       {"role": "user", "content": "A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess? Provide all solutions to that question."}
     ]
   }
   ```

To solve these problems, send a POST request to `/v1/chat/completions` with the above JSON payloads.

## How It Works

Fragaria employs a sophisticated Chain of Thought (CoT) reasoning process enhanced by Reinforcement Learning:

1. **Problem Classification**: Categorizes the input problem into known or new problem types.
2. **CoT Path Generation**: Creates multiple reasoning approaches for the problem type, influenced by past performance.
3. **Parallel Execution**: Applies each CoT path to the problem concurrently.
4. **Result Combination**: Synthesizes the results from different paths.
5. **Evaluation**: Scores the effectiveness of each approach.
6. **Reinforcement Learning Update**: Uses the evaluation scores as rewards to update the RL policy, influencing future path selections and generations.
7. **Adaptive Learning**: Updates the scoring database and RL model to improve future performance.

This RL-enhanced process allows Fragaria to not only tackle a wide range of problems but also to learn and adapt its strategies over time, becoming increasingly efficient at solving both familiar and novel problem types.

## Core Library

Fragaria's core library provides a powerful Python API for integrating Chain of Thought reasoning into your applications. The main components are:

### FragariaCore Class

The `FragariaCore` class is the primary interface for interacting with Fragaria's reasoning engine:

```python
from fragaria.core import FragariaCore

# Initialize the core with default or custom configuration
core = FragariaCore()

# Perform reasoning on a problem
result = await core.parallel_cot_reasoning("How many 'r's in strawberry?")
```

Key methods of the `FragariaCore` class include:

- `parallel_cot_reasoning(text, system_prompt)`: Main entry point that performs the complete CoT reasoning process
- `classify_or_create_problem_type(text)`: Classifies a problem or creates a new type
- `generate_cot_paths(text, problem_type)`: Generates multiple reasoning approaches
- `run_cot_path(session, text, path, problem_type, system_prompt)`: Executes a single reasoning path
- `combine_results(results, problem_type, system_prompt)`: Synthesizes results from multiple paths
- `evaluate_result(text, result, problem_type, system_prompt)`: Evaluates the quality of results
- `update_cot_scores(problem_type, paths, scores)`: Updates path scores in the database
- `select_cot_paths(problem_type, n)`: Selects reasoning paths using UCB algorithm
- `adapt_cot_path(path, problem_type, text, system_prompt)`: Adapts existing paths for new problems

### Convenience Functions

For simpler use cases, Fragaria provides convenience functions:

```python
from fragaria.core import analyze_problem

# Simple async function for analyzing problems
result = await analyze_problem("How many 'r's in strawberry?")
```

### Configuration

The core library is configured through a YAML file that specifies:

- LLM provider settings (OpenAI, Groq, Together.ai)
- Model configurations for different reasoning stages
- Database path for storing CoT path scores
- Server settings for the web API

### Database Integration

Fragaria uses SQLite to store and update scores for different reasoning paths, enabling the Reinforcement Learning component to improve over time. The database tracks:

- Problem types
- Reasoning methods
- Performance scores
- Usage statistics

## Contributing

We welcome contributions to Fragaria! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main Fragaria repository.

Please ensure your code adheres to the project's coding standards and include tests for new features.

## License

Fragaria is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use Fragaria in your research or wish to refer to it in your publications, please use the following BibTeX entry:

```bibtex
@software{fragaria2024,
  author       = {{Dipankar Sarkar}},
  title        = {Fragaria: Advanced Chain of Thought Reasoning API with Reinforcement Learning},
  year         = 2024,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/terraprompt/fragaria}},
}
```

For academic papers, you can cite Fragaria as:

Dipankar Sarkar. (2024). Fragaria: Advanced Chain of Thought Reasoning API with Reinforcement Learning [Computer software]. https://github.com/terraprompt/fragaria

---

Fragaria is maintained by the [TerraPrompt](https://github.com/terraprompt) team. For any questions or support, please open an issue on the GitHub repository.