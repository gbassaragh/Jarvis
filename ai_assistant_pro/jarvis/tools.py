"""
JARVIS Tool System

Plugin architecture for extending JARVIS with external tools and capabilities.
Supports web search, calculations, file operations, API calls, and more.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess

from ai_assistant_pro.utils.logging import get_logger

logger = get_logger("jarvis.tools")


@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class Tool(ABC):
    """Base class for JARVIS tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool"""
        pass

    def to_dict(self) -> Dict:
        """Convert tool to dictionary for LLM"""
        return {
            "name": self.name,
            "description": self.description,
        }


class Calculator(Tool):
    """Calculator tool for mathematical operations"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input should be a valid Python expression."
        )

    def execute(self, expression: str) -> ToolResult:
        """
        Execute mathematical expression

        Args:
            expression: Mathematical expression (e.g., "2 + 2", "sqrt(16)")

        Returns:
            ToolResult with calculation result
        """
        try:
            # Safe eval with limited scope
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }

            # Import math functions
            import math
            allowed_names.update({
                name: getattr(math, name)
                for name in dir(math)
                if not name.startswith("_")
            })

            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class WebSearch(Tool):
    """Web search tool using DuckDuckGo"""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information. Input should be a search query."
        )

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """
        Search the web

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            ToolResult with search results
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })

            return ToolResult(
                success=True,
                result=formatted_results,
                metadata={"query": query, "count": len(formatted_results)}
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult(
                success=False,
                result=[],
                error=str(e)
            )


class ShellCommand(Tool):
    """Execute shell commands (with safety restrictions)"""

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        super().__init__(
            name="shell",
            description="Execute safe shell commands. Limited to approved commands only."
        )

        # Default safe commands
        self.allowed_commands = allowed_commands or [
            "ls", "pwd", "date", "whoami", "echo",
            "cat", "head", "tail", "wc", "grep"
        ]

    def execute(self, command: str) -> ToolResult:
        """
        Execute shell command

        Args:
            command: Shell command to execute

        Returns:
            ToolResult with command output
        """
        # Check if command is allowed
        cmd_name = command.split()[0]

        if cmd_name not in self.allowed_commands:
            return ToolResult(
                success=False,
                result=None,
                error=f"Command '{cmd_name}' not allowed. Allowed: {self.allowed_commands}"
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            return ToolResult(
                success=result.returncode == 0,
                result=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"returncode": result.returncode}
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                result=None,
                error="Command timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class PythonREPL(Tool):
    """Python REPL for code execution"""

    def __init__(self):
        super().__init__(
            name="python",
            description="Execute Python code. Returns the output."
        )

    def execute(self, code: str) -> ToolResult:
        """
        Execute Python code

        Args:
            code: Python code to execute

        Returns:
            ToolResult with execution output
        """
        try:
            # Capture stdout
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            # Execute code
            exec(code, {"__builtins__": __builtins__})

            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            return ToolResult(
                success=True,
                result=output,
                metadata={"code": code}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class FileOperations(Tool):
    """File reading and writing operations"""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        super().__init__(
            name="file_ops",
            description="Read and write files. Operations limited to allowed directories."
        )

        self.allowed_dirs = allowed_dirs or ["/tmp", "./data"]

    def execute(self, operation: str, filepath: str, content: Optional[str] = None) -> ToolResult:
        """
        Perform file operation

        Args:
            operation: "read" or "write"
            filepath: Path to file
            content: Content to write (for write operation)

        Returns:
            ToolResult with operation result
        """
        from pathlib import Path

        # Security check
        path = Path(filepath).resolve()
        allowed = any(
            str(path).startswith(str(Path(d).resolve()))
            for d in self.allowed_dirs
        )

        if not allowed:
            return ToolResult(
                success=False,
                result=None,
                error=f"File path not in allowed directories: {self.allowed_dirs}"
            )

        try:
            if operation == "read":
                with open(path) as f:
                    result = f.read()

                return ToolResult(
                    success=True,
                    result=result,
                    metadata={"filepath": str(path), "size": len(result)}
                )

            elif operation == "write":
                if content is None:
                    return ToolResult(
                        success=False,
                        result=None,
                        error="Content required for write operation"
                    )

                with open(path, "w") as f:
                    f.write(content)

                return ToolResult(
                    success=True,
                    result=f"Wrote {len(content)} characters to {path}",
                    metadata={"filepath": str(path)}
                )

            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e)
            )


class ToolRegistry:
    """
    Registry of available tools

    Manages tool registration, discovery, and execution.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        logger.info("Tool registry initialized")

    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all available tools"""
        return [tool.to_dict() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool

        Args:
            tool_name: Name of tool to execute
            **kwargs: Arguments for tool

        Returns:
            ToolResult
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}"
            )

        logger.info(f"Executing tool: {tool_name}")
        return tool.execute(**kwargs)


class ToolUseParser:
    """
    Parse tool use from LLM output

    Recognizes patterns like:
    - [TOOL: calculator] 2 + 2 [/TOOL]
    - Tool: web_search, Query: "Python tutorials"
    """

    def __init__(self):
        # Pattern: [TOOL: name] args [/TOOL]
        self.pattern = re.compile(r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]', re.DOTALL)

    def parse(self, text: str) -> List[Dict]:
        """
        Parse tool uses from text

        Args:
            text: Text to parse

        Returns:
            List of tool use dictionaries
        """
        matches = self.pattern.findall(text)

        tool_uses = []
        for tool_name, args_text in matches:
            tool_uses.append({
                "tool": tool_name,
                "args": args_text.strip(),
            })

        return tool_uses

    def remove_tool_markers(self, text: str) -> str:
        """Remove tool markers from text"""
        return self.pattern.sub("", text).strip()


def create_default_tools() -> ToolRegistry:
    """Create registry with default tools"""
    registry = ToolRegistry()

    # Register default tools
    registry.register(Calculator())
    registry.register(WebSearch())
    registry.register(PythonREPL())
    registry.register(FileOperations())

    return registry


# Example usage
if __name__ == "__main__":
    # Create tool registry
    tools = create_default_tools()

    # List tools
    print("Available tools:")
    for tool in tools.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # Execute calculator
    result = tools.execute("calculator", expression="2 + 2 * 3")
    print(f"\nCalculator result: {result.result}")

    # Execute web search
    result = tools.execute("web_search", query="Python programming", num_results=3)
    if result.success:
        print(f"\nWeb search results:")
        for r in result.result:
            print(f"  - {r['title']}")
