# Hugging Face Space (Docker SDK) image for Dana's headless API + React frontend.
#
# Multi-stage: build the Tauri/React web bundle, then serve it (and the API)
# from a single FastAPI/Uvicorn process — dana/api/server.py auto-mounts
# frontend/dist/ at "/" when present. Replaces the old Gradio Space
# (hf_space/, deleted — see dana/ui/react_dispatch.py + dana/api/server.py).
#
# `deploy/requirements-space.txt` is a lean subset of the root
# requirements.txt: no torch/transformers/audio/desktop-GUI stacks, since
# this container only ever resolves the Mock* platform drivers
# (dana.platform.factory picks them whenever SPACE_ID is set, which HF sets
# automatically for every Space regardless of SDK).

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
WORKDIR /app

COPY deploy/requirements-space.txt ./requirements-space.txt
RUN pip install --no-cache-dir -r requirements-space.txt

COPY dana/ ./dana/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 7860
CMD ["uvicorn", "dana.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
