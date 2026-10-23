# Publishing

This package has not been uploaded to PyPI or TestPyPI. Do not run an upload command until the name decision below is resolved and you explicitly approve the upload.

Checked on 2026-10-23 against the public JSON API:

| Lookup | Result |
|---|---|
| `https://pypi.org/pypi/evalguard/json` | Project exists. Version 0.1.0. Summary: "Simple, zero-dependency validation for AI agent outputs." Owner account `jewjitsu`. Repository `yolojewjitsu/evalguard`. Import shown in its README: `evalguard`. |
| `https://pypi.org/pypi/mlpipelinecheck/json` | HTTP 404 on 2026-10-23. Rechecked before this rename. |
| `https://pypi.org/pypi/tabularguard/json` | HTTP 404 on 2026-10-23. Not used, because `mlpipelinecheck` was free. |

A 404 means the name was not registered at that moment. It does not reserve the name. Check `mlpipelinecheck` again immediately before creating a pending publisher.

This repository must not be published as `evalguard`. That would impersonate the existing project. `scripts/refuse_occupied_pypi_name.py` rejects a wheel or sdist whose distribution name normalizes to `evalguard`, and a wheel that installs a top-level `evalguard` package. The current distribution name and import package are both `mlpipelinecheck`.

## Chosen name

`[project].name` is `mlpipelinecheck`. The import package is `mlpipelinecheck`. The console command is `mlpipelinecheck`. The pending publisher's project name must be the same string. PyPI rejects a first upload when those names differ.

## What this workspace already verified

Local build, `twine check`, and a clean-install audit passed before the rename. After the rename, GitHub Actions on Python 3.10, 3.11, and 3.12 passed: https://github.com/HIMANSHUMOURYADTU/mlpipelinecheck/actions/runs/39958471299. TestPyPI has `mlpipelinecheck` 0.2.0, and a clean environment imported that version. No official PyPI upload has been made. No git tag or GitHub Release has been created.

## One-time accounts

1. Create a PyPI account at https://pypi.org/account/register/.
2. Create a separate TestPyPI account at https://test.pypi.org/account/register/. TestPyPI has its own database and may delete projects. It is for testing the upload, not for a permanent release.
3. Prefer Trusted Publishing. Do not create a token until the fallback section below is actually needed.

Official references used for this guide:

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://packaging.python.org/en/latest/guides/using-testpypi/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Build and check

From a clean checkout, with the development extra installed:

```bash
python -m pytest -q
python -m ruff check src tests benchmarks examples scripts
python -m ruff format --check src tests benchmarks examples scripts
rm -rf dist build
python -m build
python -m twine check dist/*
python scripts/refuse_occupied_pypi_name.py dist
```

On Windows PowerShell, replace `rm -rf dist build` with `Remove-Item -Recurse -Force dist, build`.

`twine check` reads the built metadata. It does not upload. The name script must print that the directory does not use the occupied name before any upload. For the `mlpipelinecheck` distributions it exits 0.

Install the wheel in a new virtual environment and run one audit before uploading. Dependencies come from PyPI. The package under test comes from the local wheel.

## TestPyPI

Do this only after the name script exits 0, and only with your own TestPyPI account.

This upload already succeeded. The project page is https://test.pypi.org/project/mlpipelinecheck/ and the files are `mlpipelinecheck-0.2.0-py3-none-any.whl` and `mlpipelinecheck-0.2.0.tar.gz`. The publisher that performed it is:

- PyPI project name: `mlpipelinecheck`
- Owner: `HIMANSHUMOURYADTU`
- Repository name: `mlpipelinecheck`
- Workflow name: `testpypi.yml`
- Environment name: `testpypi`

`.github/workflows/testpypi.yml` is manual (`workflow_dispatch` only). It publishes to `https://test.pypi.org/legacy/`. `.github/workflows/release.yml` publishes to pypi.org and must not be used for another TestPyPI trial. For a manual TestPyPI upload of a new version:

```bash
python -m twine upload --repository testpypi dist/*
```

Twine reads `~/.pypirc` or `TWINE_USERNAME` and `TWINE_PASSWORD`. Use username `__token__` and a TestPyPI token as the password. Do not commit that file. `.pypirc` is gitignored. Do not paste the token into the README, the workflow, source, or a chat.

Install from TestPyPI in a clean environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mlpipelinecheck
```

`--extra-index-url` lets pip fetch numpy, pandas, and scikit-learn from PyPI. Then run `python -m mlpipelinecheck`.

Confirm the project page at `https://test.pypi.org/project/mlpipelinecheck/`.

## Production Trusted Publishing

1. On the GitHub repository, create an environment named `pypi`. Enable required reviewers so a release waits for a maintainer. This workspace cannot create that environment.
2. On PyPI, open your account sidebar and add a pending publisher, as described at https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/.
   - Project name: the same value as `[project].name`
   - Owner: the GitHub owner
   - Repository: the repository name
   - Workflow: `release.yml`
   - Environment: `pypi`
3. A pending publisher does not reserve the name. If someone else registers it first, the pending publisher is invalidated.
4. Push the repository, including `.github/workflows/release.yml`.
5. Create a GitHub Release. The workflow listens to `release` with type `published`. It does not listen to `pull_request`, `pull_request_target`, or `workflow_dispatch`.
6. Approve the `pypi` environment when GitHub asks.
7. The build job runs tests, builds `dist/`, runs `twine check`, and runs the name script. The publish job downloads those files and runs `pypa/gh-action-pypi-publish@release/v1`. It does not build a second time. `id-token: write` is set on the publish job only.

The workflow file existing is not a publish. Nothing is on PyPI until that job succeeds and the project page shows the version.

If the first upload says the pending publisher's project name does not match the metadata, follow https://docs.pypi.org/trusted-publishers/troubleshooting/. Delete the wrongly created project if the metadata name is the one you intend, create the pending publisher again with that name, and upload a new version if a file was already accepted. PyPI does not let you replace an existing file.

## Manual token fallback

Use a token only when Trusted Publishing is unavailable.

- Create an API token scoped to the one project, not an account-wide token, from the PyPI account settings.
- Pass it through `TWINE_PASSWORD` or a `~/.pypirc` that stays outside the repository.
- Username is `__token__`.
- Revoke the token after the upload and create a new one for the next manual upload.
- Never commit a token, print it in a log, or put it in `pyproject.toml`, the wheel, or the workflow as a literal.

```bash
python -m twine upload dist/*
```

## Version and approval

This checkout is version 0.2.0 in both `pyproject.toml` and `src/mlpipelinecheck/_version.py`. Those two strings are tested against each other. 0.2.0 has not been uploaded under the name `mlpipelinecheck`. Do not upload it under the name `evalguard`.

After the rename, the first upload can stay 0.2.0 if that version has never been published under the new name. The next upload needs a new version. PyPI keeps a published file. A mistaken upload is corrected by a new release, or by yanking the bad file so installs do not select it by default. Yanking does not delete the file.

Stop here unless you have approved the upload. The command that performs a manual upload is `python -m twine upload dist/*`. The GitHub action that performs a Trusted Publishing upload is the `publish` job in `.github/workflows/release.yml`, and it runs only after a GitHub Release and an environment approval.

## After a real upload

Verify, and only then change the README install command:

- The project page shows the version you built.
- Both the wheel and the sdist are listed.
- A clean virtual environment can `pip install` that exact name from `https://pypi.org/simple`.
- `python -c "import mlpipelinecheck"` and `mlpipelinecheck --help` work.
- One classification audit from the installed package exits 0 on `examples/data/classification.csv`.

Until that install succeeds, the README must keep the source-install instructions.
