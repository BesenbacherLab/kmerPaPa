"""Development tasks."""

import os
import re
import sys
from pathlib import Path
from shutil import which
from typing import List, Optional, Pattern

import httpx
from duty import duty
from git_changelog.build import Changelog, Version
from jinja2.sandbox import SandboxedEnvironment

PY_SRC_PATHS = (Path(_) for _ in ("src", "tests", "duties.py", "docs/macros.py"))
PY_SRC_LIST = tuple(str(_) for _ in PY_SRC_PATHS)
PY_SRC = " ".join(PY_SRC_LIST)
TESTING = os.environ.get("TESTING", "0") in {"1", "true"}
CI = os.environ.get("CI", "0") in {"1", "true", "yes", ""}
WINDOWS = os.name == "nt"
PTY = not WINDOWS and not CI


@duty(pre=["check_docs", "check_dependencies"])
def check(ctx):
    """
    Check it all!

    Arguments:
        ctx: The context instance (passed automatically).
    """


@duty
def check_dependencies(ctx):
    """
    Check for vulnerabilities in dependencies.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    nofail = False
    safety = which("safety")
    if not safety:
        pipx = which("pipx")
        if pipx:
            safety = f"{pipx} run safety"
        else:
            safety = "safety"
            nofail = True
    ctx.run(
        f"poetry export -f requirements.txt --without-hashes | {safety} check --stdin --full-report",
        title="Checking dependencies",
        pty=PTY,
        nofail=nofail,
    )


@duty
def check_docs(ctx):
    """
    Check if the documentation builds correctly.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    Path("build/coverage").mkdir(parents=True, exist_ok=True)
    Path("build/coverage/index.html").touch(exist_ok=True)
    ctx.run("mkdocs build -s", title="Building documentation", pty=PTY)


@duty
def check_types(ctx):
    """
    Check that the code is correctly typed.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    ctx.run(f"mypy --config-file config/mypy.ini {PY_SRC}", title="Type-checking", pty=PTY)


@duty(silent=True)
def clean(ctx):
    """
    Delete temporary files.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    ctx.run("rm -rf .coverage*")
    ctx.run("rm -rf .mypy_cache")
    ctx.run("rm -rf .pytest_cache")
    ctx.run("rm -rf tests/.pytest_cache")
    ctx.run("rm -rf build")
    ctx.run("rm -rf dist")
    ctx.run("rm -rf pip-wheel-metadata")
    ctx.run("rm -rf site")
    ctx.run("find . -type d -name __pycache__ | xargs rm -rf")
    ctx.run("find . -name '*.rej' -delete")


@duty
def docs(ctx):
    """
    Build the documentation locally.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    ctx.run("mkdocs build", title="Building documentation")


@duty
def docs_serve(ctx, host="127.0.0.1", port=8000):
    """
    Serve the documentation (localhost:8000).

    Arguments:
        ctx: The context instance (passed automatically).
        host: The host to serve the docs from.
        port: The port to serve the docs on.
    """
    ctx.run(f"mkdocs serve -a {host}:{port}", title="Serving documentation", capture=False)


@duty
def docs_deploy(ctx):
    """
    Deploy the documentation on GitHub pages.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    ctx.run("mkdocs gh-deploy", title="Deploying documentation")


@duty
def release(ctx, version):
    """
    Release a new Python package.

    Arguments:
        ctx: The context instance (passed automatically).
        version: The new version number to use.
    """
    ctx.run(f"poetry version {version}", title=f"Bumping version in pyproject.toml to {version}", pty=PTY)
    ctx.run("git add pyproject.toml CHANGELOG.md", title="Staging files", pty=PTY)
    ctx.run(["git", "commit", "-m", f"chore: Prepare release {version}"], title="Committing changes", pty=PTY)
    ctx.run(f"git tag {version}", title="Tagging commit", pty=PTY)
    if not TESTING:
        ctx.run("git push", title="Pushing commits", pty=False)
        ctx.run("git push --tags", title="Pushing tags", pty=False)
        ctx.run("poetry build", title="Building dist/wheel", pty=PTY)
        ctx.run("poetry publish", title="Publishing version", pty=PTY)
        docs_deploy.run()  # type: ignore


@duty(silent=True)
def coverage(ctx):
    """
    Report coverage as text and HTML.

    Arguments:
        ctx: The context instance (passed automatically).
    """
    ctx.run("coverage combine .coverage-*", nofail=True)
    ctx.run("coverage report --rcfile=config/coverage.ini", capture=False)
    ctx.run("coverage html --rcfile=config/coverage.ini")


@duty
def test(ctx, match: str = ""):
    """
    Run the test suite.

    Arguments:
        ctx: The context instance (passed automatically).
        match: A pytest expression to filter selected tests.
    """
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    os.environ["COVERAGE_FILE"] = f".coverage-{py_version}"
    ctx.run(
        ["pytest", "-c", "config/pytest.ini", "-n", "auto", "-k", match, "tests"],
        title="Running tests",
        pty=PTY,
    )
