.PHONY: visual-setup visual-backend visual-frontend visual-dev visual-build

SHELL := /bin/bash
VENV?=visual/backend/.venv
PY?=$(VENV)/bin/python
UVICORN?=$(PY) -m uvicorn app:app --reload --host 0.0.0.0 --port $${API_PORT:-8888}


visual-setup:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		. $(VENV)/bin/activate && pip install -r visual/backend/requirements.txt; \
	fi

visual-backend: visual-setup
	cd visual/backend && . .venv/bin/activate && VISUAL_DATA_ROOT=$${VISUAL_DATA_ROOT:-$$PWD/../data} $(UVICORN)

visual-frontend:
	cd visual/frontend && npm install && VITE_API_BASE=$${VITE_API_BASE:-} npm run dev -- --port 8889

visual-dev:
	$(MAKE) visual-setup
	cd visual/backend && . .venv/bin/activate && VISUAL_DATA_ROOT=$${VISUAL_DATA_ROOT:-$$PWD/../data} API_PORT=8888 CORS_ORIGIN=* LOG_LEVEL=info python app.py & \
	BACK_PID=$$!; \
	cd visual/frontend && npm install && VITE_API_BASE=$${VITE_API_BASE:-} npm run dev -- --port 8889 & \
	FRONT_PID=$$!; \
	trap 'kill $$BACK_PID $$FRONT_PID' INT TERM; \
	wait $$BACK_PID $$FRONT_PID

visual-build:
	$(MAKE) visual-setup
	cd visual/frontend && npm install && npm run build
	cd visual/backend && . .venv/bin/activate && FRONTEND_DIST=$${FRONTEND_DIST:-$$PWD/../frontend/dist} VISUAL_DATA_ROOT=$${VISUAL_DATA_ROOT:-$$PWD/../data} python app.py
