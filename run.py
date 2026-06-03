import os
import uvicorn

if __name__ == "__main__":
    # Render assigns a dynamic port via environment variables. Default to 8000 locally.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)