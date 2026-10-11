# Publication checklist

- [ ] Confirm ownership and redistribution rights of the code and third-party contributions.
- [ ] Check third-party licensing obligations (including original EfficientAD code and Ultralytics).
- [ ] Review **all** copied Python/Bash files for hardcoded secrets, local server locations and environment-specific paths; automated scanning is limited.
- [ ] Inspect scripts with destructive commands before executing them.
- [ ] Ensure there are no checkpoints, videos, screenshots, datasets or result files staged.
- [ ] Inspect the category manifest and manually reassign any incorrectly classified scripts.
- [ ] Inspect `docs/IMPORTS.md` and run import/smoke tests with compatible dependencies.
- [ ] Do not claim production-quality performance, live-camera latency or runnable end-to-end reproduction without independent checks.

For a Linux checkout, Git should preserve the `code/` symlinks. On Windows, Git symlink behavior requires separate setup; use a Linux environment as originally developed.

```bash
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.csv' -o -name '*.png' -o -name '*.mp4' \) -print
find code -type l ! -exec test -e {} \; -print  # broken compatibility links (should be empty)
git status --short
```
