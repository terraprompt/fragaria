"""Example usage of the Fragaria library"""

import asyncio
from fargaria.core import analyze_problem

async def main():
    # Example 1: Simple problem
    print("Analyzing: How many 'r's in strawberry?")
    result = await analyze_problem("How many 'r's in strawberry?")
    print(f"Result: {result['result']}")
    print(f"Problem type: {result['problem_type']}")
    print()
    
    # Example 2: More complex problem with system prompt
    print("Analyzing: A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess?")
    system_prompt = "You are a helpful math assistant specialized in solving age-related puzzles."
    result = await analyze_problem(
        "A princess is as old as the prince will be when the princess is twice as old as the prince was when the princess's age was half the sum of their present age. What is the age of prince and princess?",
        system_prompt
    )
    print(f"Result: {result['result']}")
    print(f"Problem type: {result['problem_type']}")
    print()

if __name__ == "__main__":
    asyncio.run(main())