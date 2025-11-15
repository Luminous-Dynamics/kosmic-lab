#compdef make
# Zsh completion for Kosmic Lab Makefile
#
# Installation:
#   1. Add to your .zshrc:
#      fpath=(~/path/to/kosmic-lab/scripts $fpath)
#      autoload -Uz compinit && compinit
#
#   2. Or copy to /usr/local/share/zsh/site-functions/_make (system-wide):
#      sudo cp scripts/zsh_completion.zsh /usr/local/share/zsh/site-functions/_kosmic_make
#
# Usage:
#   Type "make " and press TAB to see all available targets

_kosmic_lab_make() {
    local -a targets
    local makefile="Makefile"

    if [[ -f "$makefile" ]]; then
        # Extract targets from Makefile
        targets=(${(f)"$(make -qp 2>/dev/null | \
            awk -F':' '/^[a-zA-Z0-9][^$#\/\t=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | \
            grep -v '^\.PHONY' | \
            sort -u)"})

        # Extract descriptions from Makefile comments
        local -A descriptions
        while IFS=':' read -r target desc; do
            if [[ -n "$target" && -n "$desc" ]]; then
                target=$(echo "$target" | xargs)  # trim whitespace
                desc=$(echo "$desc" | sed 's/^[[:space:]]*#[[:space:]]*//')  # extract comment
                descriptions[$target]="$desc"
            fi
        done < <(grep -E '^[a-zA-Z_-]+:.*?#' "$makefile")

        # Create completion list with descriptions
        local -a completion_list
        for target in $targets; do
            if [[ -n "${descriptions[$target]}" ]]; then
                completion_list+=("$target:${descriptions[$target]}")
            else
                completion_list+=("$target")
            fi
        done

        _describe -t targets 'make targets' completion_list
    else
        # Fallback to known targets
        local -a fallback_targets
        fallback_targets=(
            'help:Show all available targets'
            'init:Bootstrap poetry environment and pre-commit hooks'
            'lint:Run static analysis (black, flake8, mypy)'
            'test:Run unit + integration tests'
            'coverage:Generate test coverage report (HTML)'
            'format:Auto-format code with black and isort'
            'type-check:Run mypy type checking'
            'security-check:Run security scan with bandit'
            'ci-local:Run full CI pipeline locally'
            'docs:Build Sphinx documentation'
            'docs-serve:Build and serve docs locally'
            'benchmarks:Run performance benchmarks'
            'dashboard:Launch real-time interactive dashboard'
            'validate-install:Comprehensive installation validation'
            'profile:Profile performance bottlenecks'
            'check-all:Run ALL validation checks'
            'migrate-v1.1:Migrate from v1.0.0 to v1.1.0'
            'docker-build:Build Docker image'
            'docker-run:Run Kosmic Lab in Docker'
            'release-check:Pre-release validation checklist'
            'clean:Remove generated files'
        )
        _describe -t targets 'make targets' fallback_targets
    fi
}

_kosmic_lab_make "$@"
