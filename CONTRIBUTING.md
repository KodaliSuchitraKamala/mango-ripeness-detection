# Contributing to Mango Ripeness Detection

First off, thank you for considering contributing to Mango Ripeness Detection! 🌟

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Code Style Guide](#code-style-guide)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/KodaliSuchitraKamala/mango-ripeness-detection/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/KodaliSuchitraKamala/mango-ripeness-detection/issues/new). Be sure to include:
  - A clear and descriptive title
  - A description of the expected behavior and the observed behavior
  - Steps to reproduce the issue
  - Screenshots if relevant
  - Your environment details

### Suggesting Enhancements

- Open an issue with the **enhancement** label
- Describe the feature and why you believe it would be beneficial
- Include any relevant screenshots or mockups

### Your First Code Contribution

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Commit your changes: `git commit -m 'Add some amazing feature'`
7. Push to your fork: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Pull Requests

- Update the README.md with details of changes if needed
- Ensure your code follows the style guide
- Make sure all tests pass
- Reference any related issues in your PR description

## Development Setup

1. **Fork and clone** the repository
   ```bash
   git clone https://github.com/your-username/mango-ripeness-detection.git
   cd mango-ripeness-detection
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all functions and methods
- Keep lines under 120 characters
- Use docstrings for all public modules, functions, classes, and methods
- Write meaningful commit messages

## Project Structure

```
.
├── .github/                # GitHub configurations
│   └── workflows/         # GitHub Actions workflows
├── data/                  # Dataset (not version controlled)
├── docs/                  # Documentation files
├── models/                # Saved models (not version controlled)
├── src/                   # Source code
│   ├── api/               # FastAPI backend
│   ├── models/            # Model architecture and training code
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── .env.example          # Example environment variables
├── .gitignore
├── CONTRIBUTING.md
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Testing

Run tests using pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Example:
```
feat: add user authentication
fix: resolve image upload issue
docs: update API documentation
```

---

Thank you for your contribution! 🚀
