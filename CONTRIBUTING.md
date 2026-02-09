# Contributing to RLHF Pipeline

Thank you for your interest in contributing to the RLHF Pipeline! We welcome contributions from the community to help make this project better.

## Getting Started

1.  **Fork the repository** to your own GitHub account.
2.  **Clone the project** to your machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/rlhf-pipeline.git
    cd rlhf-pipeline
    ```
3.  **Install dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs the package in editable mode along with development tools like `black`, `ruff`, and `pytest`.

## Development Workflow

### 1. Create a Branch
Always work on a new branch for your changes:
```bash
git checkout -b feature/my-new-feature
```

### 2. Make Changes
Implement your feature or fix. Ensure your code is clean and readable.

### 3. Lint and Format
We use `ruff` and `black` to enforce code quality. Run these commands before committing:

```bash
# Format code
black src tests

# Lint code
ruff check src tests --fix
```

### 4. Run Tests
Ensure your changes don't break existing functionality:
```bash
pytest
```

## Pull Requests

1.  Push your branch to GitHub:
    ```bash
    git push origin feature/my-new-feature
    ```
2.  Open a Pull Request (PR) against the `main` branch.
3.  Describe your changes in detail and link to any relevant issues.

## Code Style

- Follow **PEP 8** guidelines.
- Use **Type Hints** where possible.
- Write **Docstrings** for classes and functions.

## Issues

If you find a bug or have a feature request, please open an issue on GitHub.
