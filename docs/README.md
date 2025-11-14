# Kosmic Lab Documentation

This directory contains the Sphinx-based documentation for Kosmic Lab.

## Quick Start

### Prerequisites

Install documentation dependencies:

```bash
poetry add --group dev sphinx sphinx-rtd-theme
```

### Building the Documentation

```bash
# From the docs/ directory
cd docs

# Build HTML documentation
make html

# Build PDF documentation (requires LaTeX)
make latexpdf

# Clean build artifacts
make clean

# View available targets
make help
```

### Viewing the Documentation

After building, open the documentation in your browser:

```bash
# macOS
open _build/html/index.html

# Linux
xdg-open _build/html/index.html

# Windows
start _build/html/index.html
```

Or use Python's built-in server:

```bash
cd _build/html
python -m http.server 8000
# Then navigate to http://localhost:8000
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Documentation home page
├── Makefile             # Build automation
├── README.md            # This file
├── api/                 # API reference (auto-generated)
│   ├── index.rst
│   ├── core.rst
│   ├── fre.rst
│   └── scripts.rst
├── guides/              # User guides (TODO)
│   ├── k_index.rst
│   ├── k_codex.rst
│   ├── bioelectric.rst
│   └── simulation.rst
├── _static/             # Static files (CSS, images)
└── _build/              # Generated documentation (git-ignored)
```

## Writing Documentation

### Docstring Format

Use Google-style docstrings in your Python code:

```python
def my_function(param1: int, param2: str) -> float:
    """
    Brief description of the function.

    More detailed explanation if needed. Can span multiple
    paragraphs and include examples.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)
        3.14
    """
    pass
```

### reStructuredText (RST) Syntax

Common RST patterns:

```rst
Headers
=======

Subheaders
----------

**bold text**
*italic text*
``code text``

- Bullet list item
- Another item

1. Numbered list
2. Another item

.. code-block:: python

   # Python code block
   import numpy as np

.. note::
   This is a note admonition

.. warning::
   This is a warning

.. seealso::
   Related documentation
```

### Auto-generating API Docs

The API documentation is auto-generated from docstrings using Sphinx autodoc:

```rst
.. automodule:: core.logging_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Common Tasks

### Adding a New Module

1. Write comprehensive docstrings in the module
2. Add the module to the appropriate API file (e.g., `api/core.rst`)
3. Rebuild documentation: `make html`

### Adding a New Guide

1. Create a new `.rst` file in `guides/`
2. Add it to the toctree in `index.rst`
3. Write the guide content
4. Build and verify: `make html`

### Updating Configuration

Edit `conf.py` to change:
- Theme and styling
- Extensions and plugins
- Build options
- Project metadata

### Checking for Broken Links

```bash
make linkcheck
```

### Checking Documentation Coverage

```bash
make coverage
```

This shows which modules lack documentation.

## Deployment

### GitHub Pages

To deploy to GitHub Pages:

```bash
# Build HTML docs
make html

# Copy to gh-pages branch
git checkout gh-pages
cp -r _build/html/* .
git add .
git commit -m "Update documentation"
git push origin gh-pages
```

### Read the Docs

For automatic builds on Read the Docs:

1. Create `.readthedocs.yml` in project root
2. Link repository to readthedocs.org
3. Documentation builds automatically on each commit

## Troubleshooting

### Import Errors During Build

Ensure all dependencies are installed:

```bash
poetry install --with dev
```

And that the project root is in the Python path (already configured in `conf.py`).

### Missing Dependencies

If Sphinx complains about missing extensions:

```bash
poetry add --group dev sphinx-rtd-theme sphinx-autodoc-typehints
```

### LaTeX Errors (PDF Build)

For PDF generation, install LaTeX:

```bash
# macOS
brew install mactex-no-gui

# Ubuntu/Debian
sudo apt-get install texlive-latex-extra

# Fedora
sudo dnf install texlive-scheme-full
```

### Theme Not Found

Install the ReadTheDocs theme:

```bash
poetry add --group dev sphinx-rtd-theme
```

## Best Practices

1. **Write docstrings first**: Document while coding, not after
2. **Use type hints**: They auto-generate better API docs
3. **Include examples**: Show how to use each function
4. **Keep it updated**: Rebuild docs with each significant change
5. **Test code examples**: Ensure examples actually work
6. **Cross-reference**: Use Sphinx references to link related docs

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Read the Docs](https://readthedocs.org/)

---

**Happy Documenting!** 📚✨
