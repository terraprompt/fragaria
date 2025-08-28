"""Command-line interface for Fragaria"""

import argparse
import asyncio
import sys
import json
from .core import analyze_problem

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Fragaria - Advanced Chain of Thought Reasoning CLI"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input text to analyze. If not provided, will read from stdin"
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        default="",
        help="System prompt to use for the analysis"
    )
    parser.add_argument(
        "--output-format",
        "-o",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        input_text = args.input
    else:
        input_text = sys.stdin.read().strip()
        
    if not input_text:
        print("Error: No input provided", file=sys.stderr)
        sys.exit(1)
        
    # Run the analysis
    try:
        result = asyncio.run(analyze_problem(input_text, args.system_prompt))
        
        if args.output_format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(result["result"])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def server():
    """Run the Fragaria web server"""
    from .main import run_server
    run_server()

if __name__ == "__main__":
    main()