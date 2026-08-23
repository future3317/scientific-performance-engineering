# Project execution environment

All benchmark, calibration, qualification, GPU, and evidence-producing runs
for this repository must execute on the server through `ssh dbcloud`, from
`/home/workspace/lrh/CODE/scientific-performance-engineering`, using the
project environment:

```bash
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python
```

Do not run project experiments or claim runtime evidence from a local Windows
Python/Conda environment. Local checks may be used only for editing support or
static inspection; their outputs are diagnostic and must not be mixed into
server evidence. Before any run, verify the server repository, commit, clean
worktree, interpreter path, and GPU/resource contract. Do not use Conda `base`.

The canonical server workflow is:

```bash
ssh dbcloud
cd /home/workspace/lrh/CODE/scientific-performance-engineering
git pull --rebase
/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python -m pytest -q
```

Qualification and official calibration must use the declared server resource
contract and must not pass `--allow-shared-gpu`.
