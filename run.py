import uvicorn
from config import load_config


def main():
    cfg = load_config()
    host = cfg.server.host
    port = cfg.server.port

    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
