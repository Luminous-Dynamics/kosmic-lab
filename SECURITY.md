# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Kosmic Lab seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Please do not create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send a detailed report to the maintainers via:

- **Email**: kosmic-lab-security@example.org
- **GitHub Security Advisory**: Use the "Security" tab in the repository

### 3. Include in Your Report

Please include the following information:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact if exploited
- **Reproduction**: Step-by-step instructions to reproduce
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have one
- **Your contact information**: For follow-up questions

### 4. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - **Critical**: Within 7 days
  - **High**: Within 14 days
  - **Medium**: Within 30 days
  - **Low**: Next release cycle

### 5. Disclosure Policy

- We will work with you to understand and validate the vulnerability
- We will develop and test a fix
- We will prepare a security advisory
- We will coordinate disclosure timing with you
- We will credit you (if desired) in the advisory

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**
   ```bash
   poetry update
   ```

2. **Review Logs**
   - Check `logs/` directory regularly
   - Look for suspicious activity

3. **Use Environment Variables**
   - Never commit `.env` files
   - Use `.env.example` as template
   - Keep secrets in environment variables

4. **Validate Configurations**
   ```bash
   # Use JSON schemas to validate configs
   jsonschema -i config.json schemas/experiment_config.schema.json
   ```

### For Contributors

1. **Security Checks**
   - Run `make security-check` before committing
   - Pre-commit hooks include bandit scanning
   - Review security warnings seriously

2. **Sensitive Data**
   - Never log secrets or credentials
   - Use `logging.debug()` for sensitive info (disabled in production)
   - Sanitize inputs before logging

3. **Dependencies**
   - Review dependency updates carefully
   - Check for known vulnerabilities
   - Pin critical dependencies

4. **Code Review**
   - Security-critical code requires 2+ reviewers
   - Look for:
     - SQL injection vectors
     - Command injection vectors
     - Path traversal vulnerabilities
     - Insecure deserialization
     - Weak cryptography

## Known Security Considerations

### 1. Experiment Execution

Experiments execute arbitrary Python code. In production:

- Run experiments in isolated containers
- Use resource limits (CPU, memory, disk)
- Implement timeouts
- Validate all inputs
- Sanitize file paths

### 2. Data Privacy

K-Codex records may contain:

- System information (hostname, CPU count)
- Git commit SHAs
- Experiment parameters

Ensure compliance with:
- Data protection regulations
- Institutional policies
- Participant consent (if applicable)

### 3. External Services

When using external services:

- Use HTTPS for all connections
- Validate SSL certificates
- Rotate API keys regularly
- Use least-privilege access
- Monitor for unusual activity

### 4. Holochain/Mycelix Integration

When using distributed storage:

- Verify data integrity (hashes)
- Validate signatures
- Use encryption for sensitive data
- Implement access controls
- Monitor DHT queries

## Security Tooling

We use the following tools for security:

- **bandit**: Python security linting
- **pre-commit**: Automated security checks
- **Dependabot**: Dependency vulnerability scanning (GitHub)
- **GitHub Security Advisories**: Vulnerability notifications
- **pytest**: Security-focused tests

## Acknowledgments

We thank the following security researchers for responsible disclosure:

- *None yet - be the first!*

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

**Last Updated**: 2025-11-15
**Contact**: kosmic-lab-security@example.org
