# VDE++ MCP Server

Vision Diffusion Encoder++ MCP Server for Claude Code - Parse UI mockups into structured layout trees.

## Features

- **UI Structure Understanding**: Converts hand-drawn mockups and screenshots into hierarchical layout JSON
- **Diffusion Preprocessing**: Uses controlled noise to focus on structural boundaries rather than visual details
- **Developer-Focused**: Optimized for IDE layouts, terminal interfaces, and coding UIs
- **MCP Compatible**: Seamless integration with Claude Code via Model Context Protocol

## Installation

```bash
# Clone/copy the vde-server directory
cd vde-server

# Install dependencies
pip install -r requirements.txt
```

## Usage with Claude Code

### 1. Add MCP Server Configuration

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "vde-layout-parser": {
      "command": "python",
      "args": ["path/to/vde-server/mcp_server.py"],
      "env": {}
    }
  }
}
```

### 2. Use in Claude Code

```
User: *uploads UI mockup image*
User: "Analyze this layout and generate the React components"

Claude Code will automatically:
1. Use the parse_layout tool to analyze the image
2. Get structured JSON like:
   {
     "type": "split",
     "orient": "v", 
     "ratio": 0.25,
     "left": {"type": "panel", "role": "project-tree"},
     "right": {"type": "panel", "role": "editor-or-sidebar"}
   }
3. Generate appropriate code based on the structure
```

## API Reference

### parse_layout Tool

**Input:**
- `image_b64` (required): Base64-encoded PNG/JPEG of UI mockup
- `snr_db` (optional, default -4.5): Noise level for structure extraction
- `add_noise` (optional, default true): Enable diffusion preprocessing
- `return_edges` (optional, default true): Return edge detection visualization

**Output:**
- Hierarchical layout JSON with split ratios and component roles
- Optional edge detection visualization
- Analysis summary

### Layout JSON Structure

```json
{
  "type": "split",           // "split", "panel", or "tabs"
  "orient": "v",             // "v" (vertical) or "h" (horizontal) for splits
  "ratio": 0.25,             // Split ratio (0.0 to 1.0)
  "left": { ... },           // Left/top child node
  "right": { ... },          // Right/bottom child node
  "role": "project-tree"     // Panel role: "project-tree", "console", "editor-or-sidebar"
}
```

## Optimal SNR Levels

- `-4.5 dB`: Best for most UI mockups and sketches
- `-6.0 dB`: More aggressive noise, good for very detailed images
- `-3.0 dB`: Lighter preprocessing, preserves more details

## Development

### Running Standalone

```bash
# Test the FastAPI server directly
python server.py

# Test with CLI
python vde.py --image mockup.png --snr_db -4.5 --out_json layout.json
```

### Testing MCP Integration

```bash
# Run MCP server
python mcp_server.py

# Test with MCP client (see test_client.py)
python test_client.py
```

## Examples

See the `examples/` directory for:
- Sample UI mockups
- Expected JSON outputs  
- Integration examples with popular frameworks

## License

MIT License - Feel free to use in your own projects!