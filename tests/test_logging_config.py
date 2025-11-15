"""
Tests for core.logging_config module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.logging_config import get_logger, setup_logging


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_basic_setup(self):
        """Test basic logging setup."""
        setup_logging(level="INFO")

        logger = logging.getLogger("test_logger")
        assert logger.level == logging.INFO or logging.root.level == logging.INFO

    def test_debug_level(self):
        """Test DEBUG level setup."""
        setup_logging(level="DEBUG")

        root_logger = logging.getLogger()
        # Either root or handler should have DEBUG
        has_debug = root_logger.level == logging.DEBUG or any(
            h.level == logging.DEBUG for h in root_logger.handlers
        )
        assert has_debug or root_logger.level <= logging.DEBUG

    def test_with_log_file(self, temp_log_dir: Path):
        """Test logging with file output."""
        log_file = temp_log_dir / "test.log"

        setup_logging(level="INFO", log_file=str(log_file))

        logger = logging.getLogger("test_file_logger")
        logger.info("Test message")

        # File should be created
        assert log_file.exists()

    @pytest.mark.parametrize(
        "level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_various_levels(self, level: str):
        """Test setup with various logging levels."""
        setup_logging(level=level)

        logger = logging.getLogger(f"test_{level.lower()}")
        expected_level = getattr(logging, level)

        # Logger or root should have correct level
        assert (
            logger.level == expected_level
            or logging.root.level == expected_level
            or any(h.level == expected_level for h in logging.root.handlers)
        )

    def test_colored_output(self):
        """Test colored output configuration."""
        setup_logging(level="INFO", colored=True)

        logger = logging.getLogger("test_colored")
        # Should not raise
        logger.info("Colored log message")

    def test_no_colored_output(self):
        """Test non-colored output configuration."""
        setup_logging(level="INFO", colored=False)

        logger = logging.getLogger("test_no_color")
        # Should not raise
        logger.info("Non-colored log message")


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_with_name(self):
        """Test getting logger with __name__."""
        logger = get_logger(__name__)

        assert isinstance(logger, logging.Logger)
        assert __name__ in logger.name

    def test_logger_hierarchy(self):
        """Test logger name hierarchy."""
        logger1 = get_logger("parent.child")
        logger2 = get_logger("parent")

        assert logger1.name == "parent.child"
        assert logger2.name == "parent"

    def test_logger_reuse(self):
        """Test that same logger is returned for same name."""
        logger1 = get_logger("test_reuse")
        logger2 = get_logger("test_reuse")

        assert logger1 is logger2

    def test_logger_methods_exist(self):
        """Test that logger has standard logging methods."""
        logger = get_logger("test_methods")

        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")

    def test_logger_can_log(self):
        """Test that logger can actually log messages."""
        setup_logging(level="INFO")
        logger = get_logger("test_can_log")

        # Should not raise
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_full_logging_workflow(self, temp_log_dir: Path):
        """Test complete logging workflow."""
        log_file = temp_log_dir / "workflow.log"

        # Setup logging
        setup_logging(level="DEBUG", log_file=str(log_file), colored=False)

        # Get logger and log at various levels
        logger = get_logger("test_workflow")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify file exists and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

        # Should contain some log messages
        assert "INFO" in content or "WARNING" in content or "ERROR" in content

    def test_multiple_loggers(self):
        """Test multiple loggers can coexist."""
        setup_logging(level="INFO")

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module3")

        # Should not raise
        logger1.info("Message from module 1")
        logger2.info("Message from module 2")
        logger3.info("Message from module 3")

        assert logger1.name != logger2.name
        assert logger2.name != logger3.name
