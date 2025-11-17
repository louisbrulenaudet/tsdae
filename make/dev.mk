dev: ## Start development server
	@echo "🚀 Starting development server..."
	uv run fastapi dev

prod: ## Start production server
	@echo "🚀 Starting production server..."
	uv run fastapi run --host 0.0.0.0 --port 8001

test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest

init: ## Initialize development environment
	@echo "🚀 Initializing project..."
	@if [ -d ".venv" ]; then rm -rf .venv; fi
	uv venv
	uv pip install --upgrade pip
	uv pip install -r pyproject.toml --all-extras
	uv sync

install-dev: ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	uv pip install -e ".[dev]"
	@echo "✅ Development dependencies installed successfully"

check: ## Run code quality checks
	@echo "🔍 Running code analysis..."
	uvx ruff check

format: ## Format source code
	@echo "🔧 Formatting code..."
	ruff format .
	ruff check --fix

upgrade: ## Update project dependencies
	@echo "📡 Upgrading dependencies..."
	uv lock --upgrade
	uv sync
	uv pip freeze > requirements.txt
	@echo "✅ Dependencies updated successfully"

pre-commit: ## Run pre-commit checks
	pre-commit run --all-files
