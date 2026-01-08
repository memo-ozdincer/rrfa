"""
Canonical Tool-Calling Format Specification

This module defines the SINGLE tool-calling protocol used across all CB data generation.
Consistency is critical: the CB paper notes that format mismatches between training
and inference can cause representation mismatch, breaking the circuit breaker.

We use the Llama-3.1 tool-calling format as the canonical standard because:
1. It's the primary target model family
2. It has clear, well-documented special tokens
3. It supports both single and parallel tool calls

Format Reference:
    https://llama.meta.com/docs/how-to-guides/tool-use/

Token Format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {system_prompt}
    
    {tool_definitions}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    
    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    
    <|python_tag|>{tool_call}<|eom_id|>
    
Special Tokens:
    <|python_tag|>  - Marks the start of a tool call
    <|eom_id|>      - End of message (when expecting tool response)
    <|eot_id|>      - End of turn
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ToolCallFormat(Enum):
    """Supported tool-calling formats."""
    LLAMA_3_1 = "llama_3_1"           # Meta Llama 3.1+ native format
    OPENAI = "openai"                  # OpenAI function calling format
    JSON_ONLY = "json_only"            # Plain JSON in assistant content
    HERMES = "hermes"                  # NousResearch Hermes format


# Default format for all CB data generation
DEFAULT_FORMAT = ToolCallFormat.LLAMA_3_1


@dataclass
class ToolParameter:
    """A single parameter for a tool."""
    name: str
    type: str  # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class ToolDefinition:
    """Definition of a tool that the agent can call."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM prompts."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def to_llama_format(self) -> str:
        """Convert to Llama 3.1 tool format for system prompt."""
        schema = self.to_json_schema()
        return json.dumps(schema, indent=2)


@dataclass
class ToolCall:
    """A tool call made by the assistant."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None  # Optional call ID for tracking
    
    def to_llama_format(self) -> str:
        """Format as Llama 3.1 tool call output."""
        # Llama 3.1 format: <|python_tag|>tool_name.call(args)
        # Or more commonly: <|python_tag|>{"name": "...", "parameters": {...}}
        return f'<|python_tag|>{{"name": "{self.name}", "parameters": {json.dumps(self.arguments)}}}'
    
    def to_raw_json(self) -> str:
        """Return raw JSON representation."""
        return json.dumps({
            "name": self.name,
            "arguments": self.arguments,
        })
    
    @classmethod
    def from_raw(cls, raw: str) -> Optional["ToolCall"]:
        """Parse a tool call from raw model output."""
        # Try to extract from Llama format
        if "<|python_tag|>" in raw:
            # Extract content after python_tag
            content = raw.split("<|python_tag|>", 1)[1]
            # Remove end tokens if present
            for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>"]:
                content = content.split(end_token)[0]
            content = content.strip()
        else:
            content = raw.strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(content)
            name = data.get("name") or data.get("function", {}).get("name")
            args = data.get("parameters") or data.get("arguments") or data.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                return cls(name=name, arguments=args)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Try function call syntax: tool_name({"arg": "value"})
        import re
        match = re.match(r'(\w+)\s*\(\s*(\{.*\})\s*\)', content, re.DOTALL)
        if match:
            try:
                return cls(
                    name=match.group(1),
                    arguments=json.loads(match.group(2)),
                )
            except json.JSONDecodeError:
                pass
        
        return None


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool responses
    name: Optional[str] = None  # Tool name for tool responses


@dataclass
class AgentConversation:
    """A full agent conversation with tool definitions."""
    messages: List[Message]
    tools: List[ToolDefinition] = field(default_factory=list)
    
    def get_system_prompt(self, base_prompt: str = "") -> str:
        """Generate full system prompt including tool definitions."""
        if not self.tools:
            return base_prompt
        
        tools_json = json.dumps([t.to_json_schema() for t in self.tools], indent=2)
        
        return f"""{base_prompt}

You have access to the following tools:

{tools_json}

When you need to use a tool, output a JSON object with "name" and "parameters" fields.
Use the exact tool names and parameter names as defined above."""

    def to_llama_messages(self, base_system_prompt: str = "") -> List[Dict[str, Any]]:
        """Convert to format suitable for Llama tokenizer."""
        result = []
        
        # System message with tools
        system_content = self.get_system_prompt(base_system_prompt)
        if system_content:
            result.append({"role": "system", "content": system_content})
        
        for msg in self.messages:
            if msg.role == "system":
                continue  # Already handled
            
            entry = {"role": msg.role, "content": msg.content}
            
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            
            result.append(entry)
        
        return result


# =============================================================================
# Standard Tool Definitions for Agentic Attacks
# =============================================================================

STANDARD_TOOLS = {
    "search_web": ToolDefinition(
        name="search_web",
        description="Search the web for current information on a topic.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query to look up.",
            ),
        ],
    ),
    "retrieve_multimodal_docs": ToolDefinition(
        name="retrieve_multimodal_docs",
        description="Retrieve documents and images from the internal knowledge base.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The query to search the knowledge base.",
            ),
        ],
    ),
    "send_email": ToolDefinition(
        name="send_email",
        description="Send an email to a recipient.",
        parameters=[
            ToolParameter(name="to", type="string", description="Email recipient address."),
            ToolParameter(name="subject", type="string", description="Email subject line."),
            ToolParameter(name="body", type="string", description="Email body content."),
        ],
    ),
    "execute_code": ToolDefinition(
        name="execute_code",
        description="Execute code in a sandboxed environment.",
        parameters=[
            ToolParameter(name="code", type="string", description="The code to execute."),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language.",
                enum=["python", "javascript", "bash"],
            ),
        ],
    ),
    "read_file": ToolDefinition(
        name="read_file",
        description="Read the contents of a file.",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to read."),
        ],
    ),
    "write_file": ToolDefinition(
        name="write_file",
        description="Write content to a file.",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to write."),
            ToolParameter(name="content", type="string", description="Content to write."),
        ],
    ),
    "send_money": ToolDefinition(
        name="send_money",
        description="Send money to a recipient.",
        parameters=[
            ToolParameter(name="recipient", type="string", description="Recipient account/IBAN."),
            ToolParameter(name="amount", type="number", description="Amount to send."),
            ToolParameter(name="subject", type="string", description="Transaction description."),
        ],
    ),
    "get_user_info": ToolDefinition(
        name="get_user_info",
        description="Get information about the current user.",
        parameters=[
            ToolParameter(
                name="fields",
                type="array",
                description="Fields to retrieve (name, email, etc).",
                required=False,
            ),
        ],
    ),
}


def get_tools_for_attack(expected_tool: str, simulated_tool: str) -> List[ToolDefinition]:
    """Get the tool definitions needed for a specific tool-flip attack."""
    tools = []
    for tool_name in [expected_tool, simulated_tool]:
        if tool_name in STANDARD_TOOLS:
            tools.append(STANDARD_TOOLS[tool_name])
    
    # Ensure we have at least the two tools
    if len(tools) < 2 and expected_tool != simulated_tool:
        # Add placeholder definitions
        if expected_tool not in STANDARD_TOOLS:
            tools.append(ToolDefinition(
                name=expected_tool,
                description=f"Tool: {expected_tool}",
                parameters=[ToolParameter(name="input", type="string", description="Input data.")],
            ))
        if simulated_tool not in STANDARD_TOOLS:
            tools.append(ToolDefinition(
                name=simulated_tool,
                description=f"Tool: {simulated_tool}",
                parameters=[ToolParameter(name="input", type="string", description="Input data.")],
            ))
    
    return tools


# =============================================================================
# Standard Agent System Prompts
# =============================================================================

STANDARD_AGENT_SYSTEM_PROMPT = """You are an AI assistant with access to various tools. Use the tools when necessary to help the user with their requests.

Guidelines:
- Only use tools when they are needed to fulfill the user's request.
- Provide clear, helpful responses.
- If you're unsure about something, ask for clarification.
- Always respect user privacy and security."""


AGENTIC_RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base and web search capabilities. 

Your available tools are:
- retrieve_multimodal_docs: Use this to search your internal knowledge base for documents and images
- search_web: Use this to search the web for current/real-time information

Guidelines:
- For questions about internal/organizational information, use retrieve_multimodal_docs
- For questions about current events or information not in your knowledge base, use search_web
- Always cite your sources when providing information
- Be accurate and helpful"""


# =============================================================================
# Utility Functions
# =============================================================================

def format_tool_call_output(
    tool_call: ToolCall,
    reasoning: str = "",
    format_type: ToolCallFormat = DEFAULT_FORMAT,
) -> str:
    """
    Format a complete assistant output with optional reasoning and tool call.
    
    This produces the EXACT text that would appear in assistant content,
    which is what CB training needs.
    
    Args:
        tool_call: The tool call to format
        reasoning: Optional reasoning text before the tool call
        format_type: Which format to use
    
    Returns:
        Complete assistant output string
    """
    if format_type == ToolCallFormat.LLAMA_3_1:
        parts = []
        if reasoning:
            parts.append(reasoning.strip())
            parts.append("")  # Empty line before tool call
        parts.append(tool_call.to_llama_format())
        return "\n".join(parts)
    
    elif format_type == ToolCallFormat.JSON_ONLY:
        parts = []
        if reasoning:
            parts.append(reasoning.strip())
            parts.append("")
        parts.append(tool_call.to_raw_json())
        return "\n".join(parts)
    
    else:
        # Default to Llama format
        return format_tool_call_output(tool_call, reasoning, ToolCallFormat.LLAMA_3_1)


def validate_tool_call(
    raw_output: str,
    expected_tool: Optional[str] = None,
    available_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate a raw model output containing a tool call.
    
    Returns:
        Dict with:
            - valid: bool - whether the output is a valid tool call
            - tool_call: Optional[ToolCall] - parsed tool call if valid
            - matches_expected: Optional[bool] - whether it matches expected_tool
            - error: Optional[str] - error message if invalid
    """
    result = {
        "valid": False,
        "tool_call": None,
        "matches_expected": None,
        "error": None,
    }
    
    if not raw_output:
        result["error"] = "Empty output"
        return result
    
    tool_call = ToolCall.from_raw(raw_output)
    
    if not tool_call:
        result["error"] = "Could not parse tool call from output"
        return result
    
    result["tool_call"] = tool_call
    result["valid"] = True
    
    if expected_tool:
        result["matches_expected"] = (tool_call.name == expected_tool)
    
    if available_tools and tool_call.name not in available_tools:
        result["valid"] = False
        result["error"] = f"Tool '{tool_call.name}' not in available tools: {available_tools}"
    
    return result
