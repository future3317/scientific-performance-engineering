# Main branch governance

The formal calibration and release path is only allowed to land on main
through a reviewed pull request. Repository administrators must configure
GitHub branch protection for main with these settings:

- require pull request review before merging;
- require the contract and executor-contract checks to pass;
- dismiss stale approvals when the pull request changes;
- disable force pushes and branch deletion.

This file is the repository-side contract for that setting; GitHub branch
protection is an external repository setting and is not represented by a
tracked file. A readiness report must therefore record the authenticated
GitHub API result rather than claiming that the setting is enabled from this
document alone.

The server checkout workflow must continue to use the frozen project
interpreter and must not bypass review by pushing force-updated history.
