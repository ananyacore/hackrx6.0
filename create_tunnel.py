#!/usr/bin/env python3
"""
Create a public tunnel for the FastAPI application
"""

from pyngrok import ngrok
import time

try:
    # Create a tunnel to port 8001
    print("Creating tunnel to localhost:8001...")
    tunnel = ngrok.connect(8001)
    
    # Get the public URL
    public_url = tunnel.public_url
    print(f"\n🚀 PUBLIC WEBHOOK URL: {public_url}")
    print(f"📋 Submit this URL to the hackathon website: {public_url}")
    print("\n📍 Available endpoints:")
    print(f"   • Health Check: {public_url}/health")
    print(f"   • Upload Document: {public_url}/upload")
    print(f"   • Ask Question: {public_url}/ask")
    print(f"   • API Docs: {public_url}/docs")
    
    print("\n⚠️  Keep this script running to maintain the tunnel!")
    print("   Press Ctrl+C to stop the tunnel")
    
    # Keep the tunnel alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping tunnel...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("✅ Tunnel stopped")

except Exception as e:
    print(f"❌ Error creating tunnel: {e}")
    print("\n💡 Alternative options:")
    print("1. Sign up for ngrok account at: https://dashboard.ngrok.com/signup")
    print("2. Use a cloud service like Railway, Heroku, or Render")
    print("3. Use VS Code port forwarding if available")