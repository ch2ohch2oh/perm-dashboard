set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments

dashboard host="127.0.0.1" start_port="8501":
    port={{start_port}}; \
    while lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; do \
      port=$((port + 1)); \
    done; \
    echo "Starting dashboard at http://{{host}}:$port"; \
    exec .venv/bin/streamlit run app/dashboard.py --server.address {{host}} --server.port "$port" --browser.gatherUsageStats false
