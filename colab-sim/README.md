# colab-sim — proving the labs run on a fresh runtime

Not part of the shipped course. This is the verification harness used while
building it: a container with no data-science stack pre-installed, standing
in for a fresh Colab runtime, so each lab's own `!pip install` setup cell is
what has to actually work — not whatever happens to already be on the
machine that wrote the notebook.

```bash
docker build -t tsf-colab-sim -f colab-sim/Dockerfile .
docker run --rm -v "$(pwd)":/work tsf-colab-sim python colab-sim/run_labs.py
```

Executed copies (with real captured output) land in `colab-sim/out/`
(gitignored) for inspection. A clean run means every notebook's committed
outputs in `day1/`, `day2/`, `day3/` are the real, current output of running
that exact notebook — not hand-written or carried over from an earlier
version of the code.
