#!/usr/bin/env python3
"""
Contribution Statistics Generator for Kosmic Lab

This script analyzes the git repository to generate detailed contribution
statistics, including commits, lines changed, file types, and more.

Usage:
    python scripts/contribution_stats.py [--output OUTPUT] [--format FORMAT] [--since DATE]

Options:
    --output PATH    Output file path (default: CONTRIBUTOR_STATS.md)
    --format FORMAT  Output format: markdown, json, html (default: markdown)
    --since DATE     Analyze commits since this date (YYYY-MM-DD)
    --top N          Show top N contributors (default: 20)
    --verbose        Verbose output

Examples:
    # Generate markdown statistics
    python scripts/contribution_stats.py

    # Generate JSON statistics for the last year
    python scripts/contribution_stats.py --format json --since 2024-01-01

    # Show top 10 contributors in HTML
    python scripts/contribution_stats.py --format html --top 10 --output stats.html
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Try to import optional dependencies
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class ContributionAnalyzer:
    """Analyzes git repository contributions."""

    def __init__(self, repo_path: str = ".", verbose: bool = False):
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.stats: Dict[str, Any] = {}

    def log(self, message: str) -> None:
        """Print verbose log message."""
        if self.verbose:
            print(f"[INFO] {message}")

    def run_git_command(self, args: List[str]) -> str:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path)] + args,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running git command: {e}", file=sys.stderr)
            return ""

    def analyze_contributors(self, since: str = None) -> Dict[str, Dict[str, int]]:
        """Analyze contributor statistics."""
        self.log("Analyzing contributors...")

        # Build git log command
        log_args = [
            "log",
            "--format=%aN|%aE|%h|%s",
            "--numstat"
        ]

        if since:
            log_args.append(f"--since={since}")

        output = self.run_git_command(log_args)

        contributors: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "commits": 0,
            "additions": 0,
            "deletions": 0,
            "files_changed": set(),
            "email": "",
            "commit_messages": []
        })

        current_author = None
        current_email = None
        current_message = None

        for line in output.split("\n"):
            if "|" in line and not line.startswith("\t"):
                # Commit line
                parts = line.split("|")
                if len(parts) >= 4:
                    current_author = parts[0]
                    current_email = parts[1]
                    current_message = parts[3]

                    contributors[current_author]["commits"] += 1
                    contributors[current_author]["email"] = current_email
                    contributors[current_author]["commit_messages"].append(current_message)

            elif line.startswith("\t") or line.strip().isdigit():
                # File change line
                parts = line.split("\t")
                if len(parts) == 3 and current_author:
                    additions = parts[0].strip()
                    deletions = parts[1].strip()
                    filename = parts[2].strip()

                    if additions.isdigit():
                        contributors[current_author]["additions"] += int(additions)
                    if deletions.isdigit():
                        contributors[current_author]["deletions"] += int(deletions)

                    contributors[current_author]["files_changed"].add(filename)

        # Convert sets to counts
        result = {}
        for author, data in contributors.items():
            result[author] = {
                "commits": data["commits"],
                "additions": data["additions"],
                "deletions": data["deletions"],
                "files_changed": len(data["files_changed"]),
                "total_changes": data["additions"] + data["deletions"],
                "email": data["email"]
            }

        return result

    def analyze_file_types(self, since: str = None) -> Dict[str, int]:
        """Analyze contributions by file type."""
        self.log("Analyzing file types...")

        log_args = ["log", "--format=", "--name-only"]
        if since:
            log_args.append(f"--since={since}")

        output = self.run_git_command(log_args)

        file_types: Dict[str, int] = defaultdict(int)

        for line in output.split("\n"):
            if line.strip():
                ext = Path(line).suffix or "[no extension]"
                file_types[ext] += 1

        return dict(file_types)

    def analyze_timeline(self, since: str = None) -> Dict[str, int]:
        """Analyze commits over time (by month)."""
        self.log("Analyzing timeline...")

        log_args = ["log", "--format=%ad", "--date=format:%Y-%m"]
        if since:
            log_args.append(f"--since={since}")

        output = self.run_git_command(log_args)

        timeline: Dict[str, int] = defaultdict(int)

        for line in output.split("\n"):
            if line.strip():
                timeline[line.strip()] += 1

        return dict(timeline)

    def get_repository_info(self) -> Dict[str, Any]:
        """Get general repository information."""
        self.log("Getting repository info...")

        total_commits = self.run_git_command(["rev-list", "--count", "HEAD"])
        first_commit_date = self.run_git_command(["log", "--reverse", "--format=%ai", "--max-count=1"])
        last_commit_date = self.run_git_command(["log", "--format=%ai", "--max-count=1"])
        branches = len(self.run_git_command(["branch", "-a"]).split("\n"))
        current_branch = self.run_git_command(["branch", "--show-current"])

        return {
            "total_commits": int(total_commits) if total_commits else 0,
            "first_commit": first_commit_date.split()[0] if first_commit_date else "Unknown",
            "last_commit": last_commit_date.split()[0] if last_commit_date else "Unknown",
            "branches": branches,
            "current_branch": current_branch or "Unknown"
        }

    def analyze(self, since: str = None) -> Dict[str, Any]:
        """Run complete analysis."""
        self.log(f"Starting analysis{f' since {since}' if since else ''}...")

        return {
            "repository": self.get_repository_info(),
            "contributors": self.analyze_contributors(since),
            "file_types": self.analyze_file_types(since),
            "timeline": self.analyze_timeline(since),
            "generated_at": datetime.now().isoformat()
        }


class StatsFormatter:
    """Formats contribution statistics in various formats."""

    @staticmethod
    def format_markdown(stats: Dict[str, Any], top_n: int = 20) -> str:
        """Format statistics as Markdown."""
        md_lines = [
            "# Kosmic Lab Contribution Statistics",
            "",
            f"**Generated**: {stats['generated_at']}",
            "",
            "---",
            "",
            "## Repository Overview",
            "",
            f"- **Total Commits**: {stats['repository']['total_commits']:,}",
            f"- **First Commit**: {stats['repository']['first_commit']}",
            f"- **Last Commit**: {stats['repository']['last_commit']}",
            f"- **Branches**: {stats['repository']['branches']}",
            f"- **Current Branch**: {stats['repository']['current_branch']}",
            "",
            "---",
            "",
            f"## Top {top_n} Contributors",
            "",
        ]

        # Sort contributors by total changes
        sorted_contributors = sorted(
            stats['contributors'].items(),
            key=lambda x: x[1]['total_changes'],
            reverse=True
        )[:top_n]

        # Table header
        md_lines.extend([
            "| Rank | Contributor | Commits | Files | Additions | Deletions | Total Changes |",
            "|------|-------------|---------|-------|-----------|-----------|---------------|"
        ])

        # Table rows
        for rank, (name, data) in enumerate(sorted_contributors, 1):
            md_lines.append(
                f"| {rank} | {name} | {data['commits']} | {data['files_changed']} | "
                f"+{data['additions']:,} | -{data['deletions']:,} | {data['total_changes']:,} |"
            )

        md_lines.extend(["", "---", "", "## File Type Distribution", ""])

        # Sort file types by count
        sorted_file_types = sorted(
            stats['file_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]  # Top 15 file types

        md_lines.extend([
            "| Extension | Changes |",
            "|-----------|---------|"
        ])

        for ext, count in sorted_file_types:
            md_lines.append(f"| `{ext}` | {count:,} |")

        md_lines.extend(["", "---", "", "## Activity Timeline (by Month)", ""])

        # Sort timeline chronologically
        sorted_timeline = sorted(stats['timeline'].items())

        md_lines.extend([
            "| Month | Commits |",
            "|-------|---------|"
        ])

        for month, commits in sorted_timeline[-12:]:  # Last 12 months
            md_lines.append(f"| {month} | {commits} |")

        md_lines.extend([
            "",
            "---",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Contributors**: {len(stats['contributors'])}",
            f"- **Total File Types**: {len(stats['file_types'])}",
            f"- **Most Active Month**: {max(stats['timeline'].items(), key=lambda x: x[1])[0] if stats['timeline'] else 'N/A'}",
            "",
            "---",
            "",
            "*Generated with `scripts/contribution_stats.py`*"
        ])

        return "\n".join(md_lines)

    @staticmethod
    def format_json(stats: Dict[str, Any]) -> str:
        """Format statistics as JSON."""
        # Convert sets to lists for JSON serialization
        return json.dumps(stats, indent=2, default=str)

    @staticmethod
    def format_html(stats: Dict[str, Any], top_n: int = 20) -> str:
        """Format statistics as HTML."""
        sorted_contributors = sorted(
            stats['contributors'].items(),
            key=lambda x: x[1]['total_changes'],
            reverse=True
        )[:top_n]

        contributor_rows = ""
        for rank, (name, data) in enumerate(sorted_contributors, 1):
            contributor_rows += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{name}</strong></td>
                <td>{data['commits']}</td>
                <td>{data['files_changed']}</td>
                <td style="color: green;">+{data['additions']:,}</td>
                <td style="color: red;">-{data['deletions']:,}</td>
                <td><strong>{data['total_changes']:,}</strong></td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kosmic Lab Contribution Statistics</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .stats-box {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-item {{
            margin: 10px 0;
        }}
        .stat-label {{
            font-weight: 600;
            color: #666;
        }}
        .stat-value {{
            color: #333;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <h1>🚀 Kosmic Lab Contribution Statistics</h1>

    <div class="stats-box">
        <p><strong>Generated:</strong> {stats['generated_at']}</p>
    </div>

    <div class="stats-box">
        <h2>📊 Repository Overview</h2>
        <div class="stat-item">
            <span class="stat-label">Total Commits:</span>
            <span class="stat-value">{stats['repository']['total_commits']:,}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">First Commit:</span>
            <span class="stat-value">{stats['repository']['first_commit']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Last Commit:</span>
            <span class="stat-value">{stats['repository']['last_commit']}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Contributors:</span>
            <span class="stat-value">{len(stats['contributors'])}</span>
        </div>
    </div>

    <h2>👥 Top {top_n} Contributors</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Contributor</th>
                <th>Commits</th>
                <th>Files</th>
                <th>Additions</th>
                <th>Deletions</th>
                <th>Total Changes</th>
            </tr>
        </thead>
        <tbody>
            {contributor_rows}
        </tbody>
    </table>

    <p style="text-align: center; color: #666; margin-top: 40px;">
        <em>Generated with <code>scripts/contribution_stats.py</code></em>
    </p>
</body>
</html>
        """

        return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate contribution statistics for Kosmic Lab"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="CONTRIBUTOR_STATS.md",
        help="Output file path (default: CONTRIBUTOR_STATS.md)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Analyze commits since this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show top N contributors (default: 20)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = ContributionAnalyzer(verbose=args.verbose)
    stats = analyzer.analyze(since=args.since)

    # Format output
    if args.format == "markdown":
        output = StatsFormatter.format_markdown(stats, top_n=args.top)
    elif args.format == "json":
        output = StatsFormatter.format_json(stats)
    elif args.format == "html":
        output = StatsFormatter.format_html(stats, top_n=args.top)
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)

    # Write output
    output_path = Path(args.output)
    output_path.write_text(output, encoding="utf-8")

    print(f"✅ Statistics generated: {output_path}")
    print(f"   Format: {args.format}")
    print(f"   Contributors: {len(stats['contributors'])}")
    print(f"   Total commits: {stats['repository']['total_commits']:,}")


if __name__ == "__main__":
    main()
