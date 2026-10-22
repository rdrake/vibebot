.PHONY: install run test test-all lint format format-check typecheck check check-fast preflight ci clean clean-logs clean-runtime deep-clean setup-http help \
       docker-build docker-run install-service uninstall-service install-timer uninstall-timer install-hooks pre-commit \
       install-deploy worktree-create worktree-remove wait-ci push-and-wait deploy rebase-pr docs docs-serve changelog

install:
	uv sync

install-hooks:
	uv run prek install
	@echo "Git hooks installed"

pre-commit:
	uv run prek run --all-files

run:
	uv run limnoria bot.conf

test:
	uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ -v -m "not slow" --cov --cov-report=term-missing --cov-fail-under=93

test-all:
	uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ -v --cov --cov-report=term-missing --cov-fail-under=93

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run ty check plugins/llm/src/ plugins/nickinmiddle/src/

check: lint format-check typecheck test

# Same as `check` minus the test suite — used by the pre-push hook so
# pushes stay fast and CI owns full test coverage.
check-fast: lint format-check typecheck

preflight: format check

# Local approximation of CI — the authoritative definition lives in
# .github/workflows/ci.yml (prek + tests on Python 3.14 and 3.15).
ci:
	uv sync --locked
	uv run prek run --all-files
	$(MAKE) check
	$(MAKE) test-all

# Worktree workflow
WORKTREE_DIR ?= .worktrees

worktree-create:
ifndef BRANCH
	$(error BRANCH is required: make worktree-create BRANCH=fix/my-fix)
endif
	@mkdir -p $(WORKTREE_DIR)
	git worktree add $(WORKTREE_DIR)/$(BRANCH) -b $(BRANCH)
	cd $(WORKTREE_DIR)/$(BRANCH) && uv sync
	@echo "Worktree ready at $(WORKTREE_DIR)/$(BRANCH)"

worktree-remove:
ifndef BRANCH
	$(error BRANCH is required: make worktree-remove BRANCH=fix/my-fix)
endif
	git worktree remove $(WORKTREE_DIR)/$(BRANCH)
	-git branch -d $(BRANCH)
	@echo "Worktree and branch $(BRANCH) removed"

# GitHub helpers
wait-ci:
	@RUN_ID=$$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId'); \
	echo "Watching run $$RUN_ID …"; \
	while true; do \
		STATUS=$$(gh run view "$$RUN_ID" --json status,conclusion --jq '.status + " " + .conclusion'); \
		echo "$$(date +%H:%M:%S) $$STATUS"; \
		case "$$STATUS" in \
			completed\ success) echo "CI passed ✓"; exit 0;; \
			completed\ *) echo "CI failed ✗"; exit 1;; \
		esac; \
		sleep 10; \
	done

# Push, wait for CI + Docker, then restart the production bot. The
# systemd unit's ExecStartPre runs `docker pull` so a restart picks
# up the freshly published image. VIBEBOT_HOST overrides the SSH
# target (default: vibebot@rdrake.org).
VIBEBOT_HOST ?= vibebot@rdrake.org

deploy: push-and-wait
	@echo "→ Restarting vibebot on $(VIBEBOT_HOST) …"
	@ssh $(VIBEBOT_HOST) 'systemctl --user restart vibebot && sleep 5 && systemctl --user is-active vibebot'
	@echo "vibebot restarted ✓"

# Push current branch and block until CI + Docker image build both succeed.
# Docker is workflow_run-triggered after CI, so we wait for them in series.
push-and-wait:
	@BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	echo "Pushing $$BRANCH…"; \
	git push origin "$$BRANCH"; \
	SHA=$$(git rev-parse HEAD); \
	for WF in "CI" "Build and Push Docker Image"; do \
		echo "→ Waiting for $$WF on $$SHA …"; \
		RUN_ID=""; \
		while [ -z "$$RUN_ID" ] || [ "$$RUN_ID" = "null" ]; do \
			sleep 3; \
			RUN_ID=$$(gh run list --workflow="$$WF" --commit="$$SHA" --limit 1 --json databaseId --jq '.[0].databaseId // ""'); \
		done; \
		gh run watch "$$RUN_ID" --exit-status >/dev/null || { echo "$$WF failed ✗ (run $$RUN_ID)"; exit 1; }; \
		echo "$$WF passed ✓"; \
	done; \
	echo "Image published — safe to restart."

rebase-pr:
ifndef PR
	$(error PR is required: make rebase-pr PR=42)
endif
	gh pr comment $(PR) --body "@dependabot rebase"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ty_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-logs:
	@echo "Wiping local log artifacts (gitignored)…"
	rm -f messages.log
	rm -rf logs/
	rm -rf plugins/llm/logs/
	@echo "Logs cleared."

clean-runtime: clean-logs
	@echo "Wiping local runtime artifacts (gitignored)…"
	rm -f bot.conf.bak
	rm -rf tmp/ plugins/llm/tmp/
	rm -rf backup/ plugins/llm/backup/
	@echo "Runtime artifacts cleared. (bot.conf and conf/ left untouched)"

deep-clean: clean clean-runtime
	rm -rf .venv
	uv cache clean

# Prod-host recipe (run on the server, not a dev box): the bot writes
# generated pages/images to /var/www/llm behind the reverse proxy.
setup-http:
	@echo "Creating HTTP directory for code/image output..."
	mkdir -p /var/www/llm
	chmod 755 /var/www/llm
	@echo "HTTP directory created at /var/www/llm"
	@echo "Configure your web server to serve this directory"
	@echo ""
	@echo "Example nginx config:"
	@echo "  location /llm {"
	@echo "    root /var/www;"
	@echo "    autoindex off;"
	@echo "  }"

help:
	@echo "Available targets:"
	@echo "  install         - Install dependencies with uv"
	@echo "  install-hooks   - Install git pre-commit hooks"
	@echo "  pre-commit      - Run pre-commit on all files"
	@echo "  run             - Start the bot"
	@echo "  test            - Run all tests"
	@echo "  lint            - Run ruff linter"
	@echo "  format          - Format code with ruff"
	@echo "  format-check    - Check formatting without changes"
	@echo "  typecheck       - Run ty type checker"
	@echo "  check           - Run all checks (lint, format-check, typecheck, test)"
	@echo "  check-fast      - Same as check minus tests (used by the pre-push hook)"
	@echo "  preflight       - Auto-format then run all checks"
	@echo "  ci              - Run CI checks (sync --locked, pre-commit, test with coverage)"
	@echo "  worktree-create - Create isolated worktree (BRANCH=name required)"
	@echo "  worktree-remove - Remove worktree and branch (BRANCH=name required)"
	@echo "  wait-ci         - Watch current GitHub Actions run until completion"
	@echo "  push-and-wait   - Push current branch, then wait for CI + Docker image build"
	@echo "  deploy          - push-and-wait, then restart the bot via SSH (VIBEBOT_HOST=...)"
	@echo "  rebase-pr       - Ask dependabot to rebase a PR (PR=number required)"
	@echo "  clean           - Remove cache files (__pycache__, .pytest_cache, .ruff_cache)"
	@echo "  clean-logs      - Remove local logs/ and messages.log artifacts"
	@echo "  clean-runtime   - Remove logs, tmp/, backup/, bot.conf.bak (keeps bot.conf)"
	@echo "  deep-clean      - Remove venv, uv cache, runtime artifacts (full reset)"
	@echo "  changelog       - Regenerate CHANGELOG.md from git history (requires git-cliff)"
	@echo "  setup-http      - Create HTTP directory for code/image output"
	@echo "  docs            - Build MkDocs site"
	@echo "  docs-serve      - Serve docs locally with live reload"
	@echo "  docker-build    - Build Docker image locally"
	@echo "  docker-run      - Run Docker container locally"
	@echo "  install-service - Install systemd user service"
	@echo "  uninstall-service - Remove systemd user service"
	@echo "  install-timer   - Install auto-update timer (checks GHCR every 15 min)"
	@echo "  uninstall-timer - Remove auto-update timer"
	@echo "  install-deploy  - Install service and timer together"

# Documentation
docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

# Changelog (requires `git-cliff` on PATH; brew install git-cliff)
changelog:
	@command -v git-cliff >/dev/null 2>&1 || { echo "git-cliff not installed (brew install git-cliff)"; exit 1; }
	git-cliff --output CHANGELOG.md
	@echo "CHANGELOG.md updated."

# Docker
IMAGE_NAME ?= ghcr.io/rdrake/vibebot
IMAGE_TAG ?= latest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm -it \
		--user $$(id -u):$$(id -g) \
		-v $(PWD)/bot.conf:/app/bot.conf:ro \
		-v $(PWD)/conf:/app/conf \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		$(IMAGE_NAME):$(IMAGE_TAG)

# systemd user service installation
install-service:
	@echo "Creating directories..."
	mkdir -p ~/.config/systemd/user
	mkdir -p ~/.config/vibebot
	@echo "Installing systemd unit..."
	cp vibebot.service ~/.config/systemd/user/
	@if [ ! -f ~/.config/vibebot/env ]; then \
		echo "Copying example config..."; \
		cp .env.example ~/.config/vibebot/env; \
	else \
		echo "Keeping existing env file..."; \
	fi
	@if [ ! -f ~/.config/vibebot/bot.conf ]; then \
		echo "NOTE: Copy your bot.conf to ~/.config/vibebot/bot.conf"; \
	fi
	@echo "Reloading systemd..."
	systemctl --user daemon-reload
	@echo ""
	@echo "Installation complete. Next steps:"
	@echo "  1. Copy bot.conf to ~/.config/vibebot/bot.conf"
	@echo "  2. Edit ~/.config/vibebot/env with your API keys"
	@echo "  3. systemctl --user enable vibebot"
	@echo "  4. systemctl --user start vibebot"
	@echo "  5. loginctl enable-linger $$USER  (keeps service running after logout)"

uninstall-service:
	-systemctl --user stop vibebot
	-systemctl --user disable vibebot
	rm -f ~/.config/systemd/user/vibebot.service
	systemctl --user daemon-reload
	@echo "Service removed. Config files in ~/.config/vibebot/ preserved."

install-timer:
	@echo "Installing update timer..."
	mkdir -p ~/.config/systemd/user
	cp vibebot-updater.service vibebot-updater.timer ~/.config/systemd/user/
	systemctl --user daemon-reload
	systemctl --user enable --now vibebot-updater.timer
	@echo "Timer installed. Check status with: systemctl --user status vibebot-updater.timer"

uninstall-timer:
	-systemctl --user disable --now vibebot-updater.timer
	rm -f ~/.config/systemd/user/vibebot-updater.service
	rm -f ~/.config/systemd/user/vibebot-updater.timer
	systemctl --user daemon-reload
	@echo "Timer removed."

install-deploy: install-service install-timer
	@echo ""
	@echo "Deployment complete. Bot will auto-update every 15 minutes."
