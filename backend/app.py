import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import logging
import uvicorn
from main_minimal import app


def _env_bool(name: str, default: bool = False) -> bool:
    value = (os.getenv(name, "") or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", "8002"))
        log_level = (os.getenv("UVICORN_LOG_LEVEL", "info") or "info").strip().lower()
        access_log = _env_bool("UVICORN_ACCESS_LOG", True)
        debug = _env_bool("DEBUG", False)

        if debug:
            # Raise verbosity automatically when DEBUG=1 unless explicitly overridden.
            log_level = (os.getenv("UVICORN_LOG_LEVEL", "debug") or "debug").strip().lower()

        # Configure Python root logging so library debug/info messages are emitted
        try:
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                numeric_level = logging.INFO
            logging.basicConfig(
                level=numeric_level,
                format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # Ensure uvicorn loggers reflect chosen level as well
            logging.getLogger("uvicorn").setLevel(numeric_level)
            logging.getLogger("uvicorn.error").setLevel(numeric_level)
            logging.getLogger("uvicorn.access").setLevel(numeric_level if access_log else logging.WARNING)
        except Exception:
            # Best-effort logging config; continue even if logging setup fails
            pass

        print(
            f"Starting API on 0.0.0.0:{port} | "
            f"log_level={log_level} | access_log={access_log} | debug={debug}"
        )

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level=log_level,
            access_log=access_log,
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
