import os
import sys

# Add the current directory to the path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app import app

if __name__ == "__main__":
    # Get port from environment variable or default to 7860
    port = int(os.environ.get("PORT", 7860))
    # Get server_name from environment variable or default to 0.0.0.0 for Docker
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

    print(f"Starting app on {server_name}:{port}")
    app.launch(server_name=server_name, server_port=port)
