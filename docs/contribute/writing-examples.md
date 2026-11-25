# Writing examples

Start with a physical question and a runnable calculation. Show the result,
then add the model details needed to interpret it. The {doc}`../cookbook`
collects user recipes, and this page describes how to write and publish them.

## Organize the calculation

Keep the first calculation small. Introduce batching, model reduction or
custom extensions when the experiment needs them. Use the public device,
coupling, control and solver APIs, with object references and short labels.

Group sections around an experiment. Wiring and measurement belong together
when they explain the same signal. Link separate studies when the physical
question changes. Keep code cells focused on the definition, solution or
inspection of the model. Put plotting code in a separate, collapsible block.

State units, frames, approximations and truncations where they affect the
answer. Use a component's defined parameters for the model inputs and
identify dressed quantities as results of the coupled system. Keep the
original model when you sweep, fit or reduce it.

## Guide or study

A guide teaches one workflow on a small chip and runs in about a minute or
less. A study answers one physical question from start to end, compares the
result with published data or a known limit, and can take longer. Put guides
in `docs/guides/` and studies in `docs/studies/`. Give each page one name for
its file, its address and its title.

## Show the result

Put the figure or compact numerical output below its calculation. Avoid
narrating every step with `print()`. Print each number at the accuracy of the
calculation, which the solver tolerance or a truncation sets, so that a change
below that accuracy does not change the page. A study can end with a
machine-readable record of its parameters, observables and checks for the
tests, but the record does not replace the result shown to the reader.

Use familiar physical names. Say “bath model,” “pulse schedule,” or “parameter
sweep” instead of calling everything a “recipe.”

Give each page one title that names the calculation rather than a quchip
implementation concept, and use it for the heading, the guide index and the
sidebar.

## Plot the observable

Label axes with units and state which subsystem was traced out or which state
was prepared. Show the drive envelope when it explains the response. Use equal
axis scaling for IQ plots and a logarithmic scale when the range requires it.

Use the style in `docs/_static/quchip.mplstyle`: red, blue and charcoal, with
light and dark SVG variants and PDF downloads. Every committed figure must
come from the accompanying code. Inspect the rendered page in both themes,
and check its labels, legends, figure size and collapsed code blocks.

`chip.plot_graph()` shows bare parameters by default. Use `values="dressed"`
or `values="both"` when the comparison calls for dressed observables.

## Check the calculation

Choose a check tied to the observable. For example, add local levels, refine
the time grid, compare a derivative with finite differences, or compare the
result with a known physical limit. Inspect dressed-state overlaps near
hybridization and the elimination report when you use a reduced model.

Choose tolerances from solver accuracy, convergence, an approximation scale
such as coupling over detuning, or a stated physical estimate. Do not derive
the tolerance from the residual of the run being tested.

## Publish the example

A page is either Markdown with its printed output or a Jupyter notebook.

- A Markdown page in `docs/` shows each printed result in an `Output:` block
  below the Python block that prints it. Its code runs from `docs/images`, so
  it saves each figure there by name and loads `../_static/quchip.mplstyle`.
  After a change, run `python3 tools/execute_markdown.py <page>` to refresh
  its output blocks and figures.
- A notebook keeps its Jupytext Markdown and its executed `.ipynb` in
  `examples/`. A short page in `docs/guides/` or `docs/studies/` includes the
  Markdown and links the notebook for download. The notebook loads the
  documentation style and saves its figures only when `../docs` exists, so a
  downloaded copy also runs outside the repository. After execution, run
  `python3 tools/sync_example_outputs.py` to copy textual outputs into the
  Markdown, then run it with `--check`.

Inspect the figures and run the focused example test and the documentation
build. Validation executes the Markdown pages and the guide notebooks again
and compares their printed output with the committed text, while the release suite
does the same for the study notebooks. The
{doc}`contributing guide <../contributing>` gives the execution and validation
commands.
