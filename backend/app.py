import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn
from main import app

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "8002"))
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
