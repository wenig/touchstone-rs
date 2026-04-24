# Repository Workflow

## Branch Structure

| Branch    | Purpose                          |
|-----------|----------------------------------|
| `develop` | Default and working branch       |
| `main`    | Stable, released code only       |

## Day-to-day Development

1. Branch off `develop` for your work.
2. Open a PR targeting `develop`. PRs to any other base branch (except `develop → main`) are automatically closed by CI.
3. CI runs on every PR and push to `develop`/`main`: `cargo clippy`, `cargo fmt --check`, `cargo test`.
4. Add the `run-smoketest` label to a PR to trigger the smoketest workflow against the PR's head commit.
5. Merging to `develop` triggers the benchmark workflow for any changed algorithm and updates the leaderboard.

## Releasing

1. On `develop`, bump the version in `touchstone-rs/Cargo.toml`.
2. Open a PR from `develop` → `main`.
3. Merge the PR.
4. CI automatically:
   - Runs `cargo clippy`, `cargo fmt --check`, `cargo test`.
   - Publishes `touchstone-rs` to crates.io (`cargo publish`).
   - Creates a GitHub Release tagged `v<version>` with auto-generated release notes.

## CI Workflows at a Glance

| Workflow                | Trigger                                      | What it does                                              |
|-------------------------|----------------------------------------------|-----------------------------------------------------------|
| `rust.yml`              | push/PR on `develop` or `main`               | clippy → fmt → test → publish (main only) → release      |
| `smoketest.yml`         | PR labeled `run-smoketest` targeting develop | Builds and smoke-tests a single changed algorithm         |
| `benchmark.yml`         | push to `develop` (algorithms changed)       | Builds, runs, and publishes benchmark results             |
| `enforce-base-branch.yml` | PR opened/reopened                         | Closes PRs not targeting `develop` (except `develop→main`) |
| `revoke-smoketest.yml`  | —                                            | Revokes the smoketest label after it runs                 |
| `audit.yml`             | —                                            | Dependency audit                                          |
