# PyPI Release Checklist

## 1. Local packaging validation
- [x] `pyproject.toml` parses cleanly
- [x] `pip install -e .` works in a clean venv
- [x] `python -c "from representation_intelligence import RepresentationIntelligenceEngine"` succeeds

## 2. Build artifacts
- [ ] Build wheel and sdist:
  ```bash
  python -m build
  ```
- [ ] Inspect `dist/` contents:
  ```bash
  twine check dist/*
  ```

## 3. PyPI metadata
- [ ] Confirm name: `representation-intelligence`
- [ ] Confirm version: `1.0.0`
- [ ] Confirm description and classifiers (Python 3, OS independent, audio analysis)
- [ ] Confirm README renders on PyPI
- [ ] Confirm homepage URL points to https://github.com/SpacekidLabs/representation-fragility-lab

## 4. Upload
- [ ] Upload to Test PyPI first:
  ```bash
  twine upload --repository testpypi dist/*
  ```
- [ ] Verify install from Test PyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ representation-intelligence
  ```
- [ ] Upload to production PyPI:
  ```bash
  twine upload dist/*
  ```

## 5. Post-release
- [ ] Update README badge from dev badge to PyPI badge
- [ ] Tag release in git:
  ```bash
  git tag -a v1.0.0 -m "First PyPI release"
  git push origin v1.0.0
  ```
- [ ] Announce/distribute as `pip install representation-intelligence`
