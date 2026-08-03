#!/usr/bin/env bash
# Start (or restart) the KC-135 Maintenance Analytics stack.
#
# Exists because the JupyterHub web terminal auto-indents pasted text and
# silently truncates backslash continuations — the launch command has been
# mangled four separate times, once leaving Streamlit running in the
# foreground with no log redirect. One short command cannot be truncated.
#
#   ./run_app.sh            start ollama (if needed) + restart streamlit
#   ./run_app.sh --no-llm   skip ollama entirely
#
# Env overrides:
#   WUC_MODEL_PATH    classifier checkpoint   (default ./wuc-model-hier)
#   WUC_OLLAMA_MODEL  pin a summary engine to the top of the dropdown
#   PORT              streamlit port          (default 8501)
set -u

cd "$(dirname "$0")" || exit 1

MODEL_PATH="${WUC_MODEL_PATH:-./wuc-model-hier}"
PORT="${PORT:-8501}"
VENV="$HOME/.venvs/wuc/bin"
STREAMLIT="$VENV/streamlit"

if [ ! -x "$STREAMLIT" ]; then
    echo "ERROR: $STREAMLIT not found."
    echo "The container may have been rebuilt on a new Python. Recreate it:"
    echo "  python -m venv --system-site-packages ~/.venvs/wuc"
    echo "  ~/.venvs/wuc/bin/pip install streamlit altair transformers huggingface-hub ollama anthropic"
    echo "Do NOT install torch — the image ships 2.12.0+cu130 and pip would replace it."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: WUC_MODEL_PATH '$MODEL_PATH' does not exist."
    echo "Without a valid checkpoint the app cannot load the classifier."
    exit 1
fi

# ---- Ollama -----------------------------------------------------------------
# KEEP_ALIVE=-1 pins the model in VRAM. Cold start is ~150s for a 10GB model and
# minutes for a 31B; without this it unloads after 5 idle minutes and every
# coffee break costs it again. The box has 48GB, so there is room to spare.
if [ "${1:-}" != "--no-llm" ]; then
    if pgrep -f "ollama serve" > /dev/null; then
        echo "ollama: already running (leaving it alone)"
    else
        echo "ollama: starting with KEEP_ALIVE=-1, FLASH_ATTENTION=1"
        OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 \
            nohup ollama serve > "$HOME/ollama.log" 2>&1 &
        sleep 2
    fi
fi

# ---- Streamlit --------------------------------------------------------------
pkill -9 -f "streamlit run main_app.py" 2>/dev/null
sleep 2

# baseUrlPath is deliberately NOT set: jupyter_server_proxy 4.x strips the
# prefix before forwarding, so setting it breaks routing behind the proxy.
WUC_MODEL_PATH="$MODEL_PATH" nohup "$STREAMLIT" run main_app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    > "$HOME/streamlit.log" 2>&1 &

sleep 6
if pgrep -f "streamlit run main_app.py" > /dev/null; then
    echo "streamlit: up on port $PORT  (model: $MODEL_PATH)"
    echo "  https://icsarl.westpoint.edu/jupyter-cdas2/user/jonathan.day/proxy/$PORT/"
    echo "  logs: tail -f ~/streamlit.log"
else
    echo "streamlit: FAILED TO START — last 20 log lines:"
    tail -20 "$HOME/streamlit.log"
    exit 1
fi
