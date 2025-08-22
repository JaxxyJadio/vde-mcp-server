#!/usr/bin/env python3
"""
MCP Server for Vision Diffusion Encoder++
Provides UI layout parsing capabilities to Claude Code via MCP protocol
"""

import asyncio
import base64
import json
import sys
from typing import Any, Dict, List
from mcp.server.models import InitializeResult
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types
from server import pipe, pil_from_b64, tensor_from_pil, b64_png

# Create MCP server instance
server = Server("vde-layout-parser")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools for the MCP server.
    Returns tools that can parse UI layouts from images.
    """
    return [
        Tool(
            name="parse_layout",
            description="Parse a raster UI mockup into a hierarchical layout tree using Vision Diffusion Encoder++. Designed specifically for developer interfaces (IDEs, terminals, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_b64": {
                        "type": "string",
                        "description": "Base64-encoded PNG or JPEG image of UI mockup or screenshot"
                    },
                    "snr_db": {
                        "type": "number",
                        "default": -4.5,
                        "description": "Target SNR in dB for diffusion preprocessing. -4.5 to -6.0 works best for UI structure extraction"
                    },
                    "add_noise": {
                        "type": "boolean", 
                        "default": True,
                        "description": "Whether to apply diffusion noise preprocessing for structure focus"
                    },
                    "return_edges": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to return edge detection visualization"
                    }
                },
                "required": ["image_b64"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.Content]:
    """
    Handle tool calls for layout parsing.
    """
    if name == "parse_layout":
        return await parse_layout_tool(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def parse_layout_tool(arguments: Dict[str, Any]) -> List[types.Content]:
    """
    Parse UI layout from base64 image using VDE++
    """
    try:
        # Extract arguments
        image_b64 = arguments.get("image_b64")
        snr_db = arguments.get("snr_db", -4.5)
        add_noise = arguments.get("add_noise", True)
        return_edges = arguments.get("return_edges", True)
        
        if not image_b64:
            raise ValueError("image_b64 is required")
        
        if pipe is None:
            raise RuntimeError("VDE Pipeline not initialized")
        
        # Convert base64 to tensor
        pil_img = pil_from_b64(image_b64)
        tensor_img = tensor_from_pil(pil_img)
        
        # Run VDE pipeline
        result = pipe.encode(tensor_img, snr_db=snr_db, add_noise=add_noise)
        
        # Prepare response content
        content = []
        
        # Add layout JSON as main content
        layout_json = json.dumps(result["layout"], indent=2)
        content.append(TextContent(
            type="text",
            text=f"""Layout Analysis Results:

```json
{layout_json}
```

Analysis Summary:
- Noise level: {snr_db} dB SNR
- Preprocessing: {'Enabled' if add_noise else 'Disabled'}
- Detected structure: {_analyze_layout_summary(result["layout"])}
"""
        ))
        
        # Add edge visualization if requested
        if return_edges and "edges" in result:
            edges = result["edges"]
            if len(edges.shape) == 2:
                edges_rgb = np.stack([edges, edges, edges], axis=-1)
            else:
                edges_rgb = edges
            edges_b64 = b64_png(edges_rgb)
            
            content.append(TextContent(
                type="text", 
                text="Edge Detection Visualization (shows structural boundaries detected):"
            ))
            
            # Note: ImageContent would be ideal here but sticking to TextContent for compatibility
            content.append(TextContent(
                type="text",
                text=f"Edge detection image (base64): data:image/png;base64,{edges_b64}"
            ))
        
        return content
        
    except Exception as e:
        import traceback
        error_msg = f"Layout parsing failed: {e}\n\nTraceback:\n{traceback.format_exc()}"
        return [TextContent(type="text", text=error_msg)]

def _analyze_layout_summary(layout: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the detected layout"""
    def count_nodes(node, node_types=None):
        if node_types is None:
            node_types = {"split": 0, "panel": 0, "tabs": 0}
        
        node_type = node.get("type", "unknown")
        if node_type in node_types:
            node_types[node_type] += 1
        
        # Recursively count child nodes
        if "left" in node and node["left"]:
            count_nodes(node["left"], node_types)
        if "right" in node and node["right"]:
            count_nodes(node["right"], node_types)
            
        return node_types
    
    counts = count_nodes(layout)
    parts = []
    
    if counts["split"] > 0:
        parts.append(f"{counts['split']} splits")
    if counts["panel"] > 0:
        parts.append(f"{counts['panel']} panels") 
    if counts["tabs"] > 0:
        parts.append(f"{counts['tabs']} tab groups")
    
    return ", ".join(parts) if parts else "single panel"

async def main():
    """Main entry point for the MCP server"""
    # Use stdin/stdout for MCP communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import mcp.server.stdio
    import numpy as np
    asyncio.run(main())