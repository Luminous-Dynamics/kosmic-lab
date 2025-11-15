#!/bin/bash
# Bash completion for Kosmic Lab Makefile
#
# Installation:
#   1. Source this file in your .bashrc or .bash_profile:
#      source /path/to/kosmic-lab/scripts/bash_completion.sh
#
#   2. Or copy to /etc/bash_completion.d/ (system-wide):
#      sudo cp scripts/bash_completion.sh /etc/bash_completion.d/kosmic-lab
#
# Usage:
#   Type "make " and press TAB to see all available targets

_kosmic_lab_make_completion() {
    local cur prev targets
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Get Makefile targets
    if [ -f "Makefile" ]; then
        targets=$(make -qp 2>/dev/null | \
            awk -F':' '/^[a-zA-Z0-9][^$#\/\t=]*:([^=]|$)/ {split($1,A,/ /);for(i in A)print A[i]}' | \
            grep -v '^\.PHONY' | \
            sort -u)
    else
        # Fallback to known targets if no Makefile
        targets="help init lint test fre-run historical-run docs docs-serve docs-clean \
                 dashboard notebook ai-suggest coverage demo clean benchmarks \
                 format type-check security-check ci-local review-improvements \
                 validate-install profile check-all migrate-v1.1 update-deps \
                 holochain-publish holochain-query holochain-verify mycelix-demo \
                 watch-tests install-dev docker-build docker-run docker-shell release-check"
    fi

    # Generate completions
    COMPREPLY=( $(compgen -W "${targets}" -- ${cur}) )
    return 0
}

# Register completion for make command in kosmic-lab directory
complete -F _kosmic_lab_make_completion make

# Also support "m" as alias for "make" (if used)
complete -F _kosmic_lab_make_completion m

echo "✅ Kosmic Lab bash completion loaded!"
echo "   Try: make [TAB][TAB]"
