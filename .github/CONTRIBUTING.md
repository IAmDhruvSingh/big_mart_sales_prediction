# Contributing to Big Mart Sales Prediction

Thank you for your interest in contributing to this project! We welcome contributions from everyone. Please follow these guidelines to ensure a smooth collaboration.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search to see if the bug has already been reported
2. **Use the Bug Report template** - Click "New issue" and select the bug template
3. **Provide detailed information:**
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Python version and key dependency versions
   - Error logs or stack traces
   - Operating system

### Requesting Features

1. **Use the Feature Request template** - Clearly describe the use case
2. **Explain the problem being solved**
3. **Suggest alternatives** you've considered
4. **Indicate priority** - is this a nice-to-have or critical capability?

### Making Code Changes

#### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/big_mart_sales_prediction.git
cd big_mart_sales_prediction
```

#### 2. Create a Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install flake8 mypy pytest pytest-cov
```

#### 4. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/documentation-update
```

#### 5. Make Your Changes

- Follow the project code style (PEP 8)
- Add docstrings to functions and classes
- Update comments if logic changes
- Write tests for new functionality
- Update `requirements.txt` if adding dependencies

#### 6. Run Quality Checks Locally

```bash
# Format code
black src/ --line-length 100

# Lint code
flake8 src/ --max-line-length=100

# Type checking
mypy src/ --ignore-missing-imports

# Run tests
pytest tests/ -v --cov=src
```

#### 7. Validate the Pipeline

Test that your changes don't break the full workflow:

```bash
# Test data processing
python -c "from src.data_processing import *; print('✓ Data processing valid')"

# Test model functionality
python -c "from src.model import *; print('✓ Model module valid')"

# Run predictions with sample data
python src/predict.py
```

#### 8. Commit with Clear Messages

```bash
git add .
git commit -m "feat: add new feature description"
# or
git commit -m "fix: resolve issue with data preprocessing"
# or
git commit -m "docs: update README with new examples"
```

**Commit message prefixes:**

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation updates
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring without feature changes
- `test:` - Test additions or updates
- `chore:` - Build, dependencies, etc.

#### 9. Push and Open a Pull Request

```bash
git push origin feature/your-feature-name
```

Then on GitHub:

1. Open a Pull Request from your branch to `main`
2. Fill out the PR template completely
3. Ensure CI checks pass
4. Request review from maintainers

## Code Style Guide

### Python Standards

- Follow **PEP 8** - Use tools like `black` and `flake8`
- **Line length:** Max 100 characters
- **Naming:**
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Documentation

- Add docstrings to all functions and classes (Google style):
  ```python
  def load_data(filepath: str) -> pd.DataFrame:
      """Load and validate training data.

      Args:
          filepath: Path to the CSV file

      Returns:
          DataFrame containing the loaded data

      Raises:
          FileNotFoundError: If filepath doesn't exist
      """
  ```

### Project Structure

Keep code organized:

- `src/data_processing.py` - Data loading and preprocessing
- `src/features.py` - Feature engineering
- `src/model.py` - Model training and evaluation
- `src/predict.py` - Prediction pipeline
- `notebooks/` - Exploratory notebooks
- `tests/` - Unit tests (if added)

## Pull Request Process

1. **Update Documentation**
   - Update README.md if adding new features
   - Add docstrings to new functions
   - Update this file if process changes

2. **Add or Update Tests**
   - Write tests for new functionality
   - Ensure all tests pass locally
   - Aim for >80% code coverage

3. **Ensure CI Passes**
   - GitHub Actions runs automatically on PR
   - All checks must pass before merge
   - Fix any linting or test failures

4. **Request Review**
   - Assign at least one reviewer
   - Address feedback promptly
   - Respond to all comments

5. **Merge**
   - Squash commits if requested
   - Delete feature branch after merge

## Development Tips

### Running the Full Pipeline

```bash
# Preprocess data
python src/data_processing.py

# Train model
python src/model.py

# Make predictions
python src/predict.py
```

### Debugging

- Add print statements or use `pdb`
- Use Jupyter notebooks for interactive exploration
- Check logs for error messages
- Test with sample data first

### Performance Considerations

- Profile code with large datasets
- Document performance impacts in PR
- Consider memory usage for big data

## Questions or Need Help?

- 📖 Check the [README](README.md) for setup instructions
- 📓 Review notebooks in `notebooks/` for examples
- 💬 Open a Discussion for questions
- 📧 Contact maintainers if needed

---

**Thank you for contributing!** Your efforts help make this project better for everyone. 🙏

---

## Recognition

Contributors will be acknowledged in:

- Project README
- Release notes for features/fixes
- GitHub contributors page
