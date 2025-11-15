# Release Checklist

This checklist ensures consistent, high-quality releases of Kosmic Lab.

**Current Version**: 1.1.0
**Last Updated**: 2025-11-15

---

## Pre-Release (1-2 weeks before)

### Code Quality
- [ ] All CI/CD checks passing on main branch
- [ ] Run full test suite locally: `make test`
- [ ] Run integration tests: `poetry run pytest tests/test_integration_end_to_end.py`
- [ ] Run code quality checks: `./scripts/check_code_quality.sh`
- [ ] Run security scan: `poetry run bandit -r core/ fre/`
- [ ] Check code coverage: Coverage ≥90%
- [ ] No critical/high severity vulnerabilities
- [ ] Review and merge all pending Dependabot PRs

### Documentation
- [ ] Update CHANGELOG.md with all changes since last release
- [ ] Update version numbers in:
  - [ ] `pyproject.toml`
  - [ ] `CHANGELOG.md`
  - [ ] `README.md`
  - [ ] `docs/conf.py`
  - [ ] `DEPLOYMENT.md`
  - [ ] `FAQ.md`
- [ ] Verify all documentation links work
- [ ] Build docs: `make docs`
- [ ] Review docs for accuracy
- [ ] Update screenshots/examples if needed

### Dependencies
- [ ] Update dependencies: `poetry update`
- [ ] Test with updated dependencies
- [ ] Lock dependencies: `poetry lock`
- [ ] Review dependency licenses (ensure compatibility)

### Testing
- [ ] Run all examples successfully:
  - [ ] `examples/01_hello_kosmic.py`
  - [ ] `examples/02_advanced_k_index.py`
  - [ ] `examples/03_multi_universe.py`
  - [ ] `examples/04_bioelectric_rescue.py`
- [ ] Run benchmarks: `make benchmarks`
- [ ] Verify performance targets met
- [ ] Test on all supported Python versions (3.10, 3.11, 3.12)
- [ ] Test Docker build: `docker build -t kosmic-lab:test .`
- [ ] Test Docker Compose: `docker-compose up`

---

## Release Preparation (1 week before)

### Version Bump
- [ ] Decide version number (following [Semantic Versioning](https://semver.org/)):
  - **Major** (x.0.0): Breaking changes
  - **Minor** (1.x.0): New features, backward compatible
  - **Patch** (1.1.x): Bug fixes, backward compatible

- [ ] Update version in `pyproject.toml`:
  ```toml
  [tool.poetry]
  version = "x.y.z"
  ```

- [ ] Update CHANGELOG.md:
  - Move items from `[Unreleased]` to new version section
  - Add release date
  - Update version history table
  - Add upgrade notes

### CHANGELOG Template

```markdown
## [x.y.z] - YYYY-MM-DD - "Release Name"

### Added
- Feature 1
- Feature 2

### Changed
- Change 1
- Change 2

### Fixed
- Fix 1
- Fix 2

### Deprecated
- Deprecated 1

### Removed
- Removed 1

### Security
- Security fix 1
```

### Git
- [ ] Create release branch: `git checkout -b release/vx.y.z`
- [ ] Commit version bump: `git commit -m "Bump version to x.y.z"`
- [ ] Push branch: `git push origin release/vx.y.z`

### Code Freeze
- [ ] Announce code freeze to team
- [ ] Only critical bug fixes allowed
- [ ] All features for this release must be merged

---

## Release Day

### Final Verification
- [ ] Pull latest from release branch
- [ ] Run full test suite: `make test` (must pass 100%)
- [ ] Run code quality: `./scripts/check_code_quality.sh` (must pass)
- [ ] Run benchmarks: `make benchmarks` (verify performance)
- [ ] Build docs: `make docs` (no errors)
- [ ] Test Docker build: `docker build -t kosmic-lab:x.y.z .`
- [ ] Verify all examples run

### Create Release
- [ ] Merge release branch to main:
  ```bash
  git checkout main
  git merge release/vx.y.z
  ```

- [ ] Create git tag:
  ```bash
  git tag -a vx.y.z -m "Release version x.y.z"
  ```

- [ ] Push to GitHub:
  ```bash
  git push origin main
  git push origin vx.y.z
  ```

### GitHub Release
- [ ] Go to https://github.com/Luminous-Dynamics/kosmic-lab/releases/new
- [ ] Select tag: `vx.y.z`
- [ ] Release title: `v x.y.z - Release Name`
- [ ] Description (use template below)
- [ ] Attach built artifacts (if any)
- [ ] Mark as pre-release if beta/rc
- [ ] Publish release

### GitHub Release Template

```markdown
# Kosmic Lab vx.y.z - "Release Name"

**Release Date**: YYYY-MM-DD
**Status**: Production Ready

## 🎉 Highlights

- Key feature 1
- Key feature 2
- Key improvement 3

## 📦 What's New

### Added
- Feature 1 (#PR-number)
- Feature 2 (#PR-number)

### Changed
- Change 1 (#PR-number)

### Fixed
- Fix 1 (#PR-number)

## 📊 Metrics

- **Tests**: XX passing
- **Coverage**: XX%
- **Documentation**: X,XXX lines
- **Performance**: [benchmark results]

## 📚 Documentation

- [CHANGELOG](https://github.com/Luminous-Dynamics/kosmic-lab/blob/main/CHANGELOG.md)
- [Documentation](https://github.com/Luminous-Dynamics/kosmic-lab/tree/main/docs)
- [FAQ](https://github.com/Luminous-Dynamics/kosmic-lab/blob/main/FAQ.md)
- [Deployment Guide](https://github.com/Luminous-Dynamics/kosmic-lab/blob/main/DEPLOYMENT.md)

## 🚀 Upgrade Instructions

See [CHANGELOG.md](https://github.com/Luminous-Dynamics/kosmic-lab/blob/main/CHANGELOG.md#upgrade-notes) for upgrade instructions.

## 🐛 Bug Reports

Report issues at: https://github.com/Luminous-Dynamics/kosmic-lab/issues

## 🙏 Contributors

Thanks to all contributors who made this release possible!
```

### Docker Release
- [ ] Build and tag Docker image:
  ```bash
  docker build -t kosmic-lab:x.y.z .
  docker tag kosmic-lab:x.y.z kosmic-lab:latest
  ```

- [ ] Push to Docker Hub (if public):
  ```bash
  docker push kosmic-lab:x.y.z
  docker push kosmic-lab:latest
  ```

### PyPI Release (if applicable)
- [ ] Build package:
  ```bash
  poetry build
  ```

- [ ] Test on TestPyPI:
  ```bash
  poetry publish -r testpypi
  ```

- [ ] Verify installation from TestPyPI:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ kosmic-lab==x.y.z
  ```

- [ ] Publish to PyPI:
  ```bash
  poetry publish
  ```

---

## Post-Release (Same Day)

### Announcements
- [ ] Tweet/social media announcement
- [ ] Post to GitHub Discussions
- [ ] Update project website (if applicable)
- [ ] Email announcement to mailing list (if applicable)
- [ ] Post to relevant communities (Reddit, HN, etc.)

### Documentation
- [ ] Verify docs build on Read the Docs (if configured)
- [ ] Update main README.md badges if needed
- [ ] Close GitHub milestone for this release

### Monitoring
- [ ] Monitor GitHub Issues for release-related bugs
- [ ] Monitor CI/CD for any failures
- [ ] Check download/clone metrics
- [ ] Review initial user feedback

---

## Post-Release (1 Week After)

### Retrospective
- [ ] Collect team feedback on release process
- [ ] Document lessons learned
- [ ] Update this checklist with improvements
- [ ] Review and close related issues

### Metrics
- [ ] Track adoption metrics:
  - GitHub stars/forks
  - Docker pulls
  - PyPI downloads (if applicable)
  - Documentation views
- [ ] Monitor error rates/bug reports
- [ ] Gather user feedback

### Next Release Planning
- [ ] Create milestone for next release
- [ ] Plan features for next release
- [ ] Update roadmap
- [ ] Create `[Unreleased]` section in CHANGELOG.md

---

## Hotfix Release Process

For critical bugs that need immediate release:

### Preparation
- [ ] Create hotfix branch from latest tag:
  ```bash
  git checkout -b hotfix/vx.y.z+1 vx.y.z
  ```
- [ ] Fix the bug
- [ ] Add tests for the fix
- [ ] Update CHANGELOG.md

### Release
- [ ] Run abbreviated tests (critical paths only)
- [ ] Bump patch version
- [ ] Merge to main
- [ ] Create tag
- [ ] Create GitHub release (mark as hotfix)
- [ ] Announce to users

---

## Version Numbering Guide

Following [Semantic Versioning 2.0.0](https://semver.org/):

### Major Version (x.0.0)
**Increment when**: Making incompatible API changes

**Examples**:
- Removing public APIs
- Changing K-Codex schema (breaking)
- Changing K-Index calculation (breaking)
- Removing Python 3.10 support

### Minor Version (1.x.0)
**Increment when**: Adding functionality in a backward compatible manner

**Examples**:
- Adding new metrics (K-Diversity, K-Harmony)
- Adding new experiment tracks
- New features (e.g., Mycelix integration)
- New helper scripts
- Enhanced documentation

### Patch Version (1.1.x)
**Increment when**: Making backward compatible bug fixes

**Examples**:
- Fixing calculation errors
- Fixing type hints
- Documentation fixes
- Dependency updates (non-breaking)
- Performance improvements

---

## Common Issues and Solutions

### Issue: CI fails on release branch
**Solution**: Fix issues on release branch, don't merge broken code to main

### Issue: Docker build fails
**Solution**: Test Docker build before creating release

### Issue: Tests fail on specific Python version
**Solution**: Fix compatibility, test all supported versions (3.10, 3.11, 3.12)

### Issue: Documentation links broken
**Solution**: Use automated link checker, verify manually

### Issue: Forgot to update version somewhere
**Solution**: Use search: `git grep "1\.1\.0"` to find all occurrences

---

## Automation Opportunities

Future improvements to automate release process:

- [ ] Automated version bumping script
- [ ] Automated CHANGELOG generation from commit messages
- [ ] Automated GitHub release creation
- [ ] Automated Docker builds on tags
- [ ] Automated PyPI publishing
- [ ] Automated announcement posting

---

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [Poetry Publishing](https://python-poetry.org/docs/libraries#publishing-to-pypi)

---

## Appendix: Release History

| Version | Date | Type | Highlights |
|---------|------|------|------------|
| 1.1.0 | 2025-11-15 | Minor | Production validation, security, community |
| 1.0.0 | 2025-11-14 | Major | First production release |
| 0.1.0 | 2025-11-09 | Minor | Publication-ready results |
| 0.0.1 | 2025-01-01 | Initial | Initial implementation |

---

*This checklist is a living document. Update it based on lessons learned from each release.*
