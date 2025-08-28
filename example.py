"""Example usage of the Fragaria library

NOTE: Before running this example, you must configure your API keys in the config.yaml file.
See the README.md for instructions on how to set up the configuration file.
"""

import asyncio
from fargaria import analyze_problem

async def main():
    # Example 1: Simple problem
    print("Analyzing: How many 'r's in strawberry?")
    try:
        result = await analyze_problem("How many 'r's in strawberry?")
        print(f"Result: {result['result']}")
        print(f"Problem type: {result['problem_type']}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please update your config.yaml file with valid API keys.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    print()
    
    # Example 2: More complex problem with system prompt
    print("Analyzing: A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess?")
    system_prompt = "You are a helpful math assistant specialized in solving age-related puzzles."
    try:
        result = await analyze_problem(
            "A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess?",
            system_prompt
        )
        print(f"Result: {result['result']}")
        print(f"Problem type: {result['problem_type']}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please update your config.yaml file with valid API keys.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    print()

if __name__ == "__main__":
    asyncio.run(main())