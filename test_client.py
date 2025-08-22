#!/usr/bin/env python3
"""
Test client for VDE++ MCP Server
Tests the layout parsing functionality with sample images
"""

import base64
import json
import requests
from PIL import Image, ImageDraw
import numpy as np

def create_test_ui_image(width=800, height=600):
    """Create a simple test UI mockup"""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a typical IDE layout
    # Left sidebar (project tree)
    draw.rectangle([0, 0, 200, height], outline='black', width=2)
    draw.text((10, 10), "Project Tree", fill='black')
    
    # Main editor area  
    draw.rectangle([200, 0, width, height-120], outline='black', width=2)
    draw.text((210, 10), "Editor Area", fill='black')
    
    # Bottom terminal
    draw.rectangle([200, height-120, width, height], outline='black', width=2)
    draw.text((210, height-110), "Terminal", fill='black')
    
    return img

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    import io
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

def test_fastapi_server():
    """Test the FastAPI server directly"""
    print("Testing FastAPI server...")
    
    # Create test image
    test_img = create_test_ui_image()
    img_b64 = image_to_base64(test_img)
    
    # Test payload
    payload = {
        "image_b64": img_b64,
        "snr_db": -4.5,
        "add_noise": True,
        "return_edges": True
    }
    
    try:
        # Test health endpoint
        health_resp = requests.get("http://localhost:8000/health")
        print(f"Health check: {health_resp.json()}")
        
        # Test layout parsing
        response = requests.post("http://localhost:8000/tools/parse_layout", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Layout parsing successful!")
            print(f"Layout: {json.dumps(result['layout'], indent=2)}")
            
            if result.get('edges_b64'):
                print("SUCCESS: Edge detection included")
            
        else:
            print(f"ERROR: Request failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Make sure to run: uvicorn server:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")

def test_mcp_protocol():
    """Test MCP protocol compliance (requires mcp client)"""
    print("\nTesting MCP protocol...")
    print("Note: This requires running the MCP server separately")
    print("Run: python mcp_server.py")
    
    # This would require an actual MCP client implementation
    # For now, just verify the server can start
    try:
        import mcp.server.stdio
        print("SUCCESS: MCP dependencies available")
    except ImportError:
        print("ERROR: MCP dependencies not found. Install with: pip install mcp")

def main():
    """Run all tests"""
    print("VDE++ MCP Server Test Suite")
    print("=" * 40)
    
    # Create and save test image
    test_img = create_test_ui_image()
    test_img.save("test_ui_mockup.png")
    print("Created test_ui_mockup.png")
    
    # Test FastAPI server (requires server to be running)
    test_fastapi_server()
    
    # Test MCP protocol
    test_mcp_protocol()
    
    print("\n" + "=" * 40)
    print("Test complete! To run full integration test:")
    print("1. Start server: uvicorn server:app --reload") 
    print("2. Run this test again")
    print("3. Test MCP: python mcp_server.py")

if __name__ == "__main__":
    main()