# Writing examples

Start with a physical question and a runnable calculation. Show the result,
then add the model details needed to interpret it. The {doc}`../cookbook`
collects user recipes. This page describes how to write and publish them.

## Organize the calculation

Keep the first calculation small. Introduce batching, model reduction or
custom extensions when the experiment needs them. Use the public device,
coupling, control and solver APIs, with object references and short labels.

Group sections around an experiment. Wiring and measurement belong together
when they explain the same signal. Link separate studies when the physical
question changes. Keep code cells focused on the declaration, solution or
inspection of the model. Put plotting code in a separate, collapsible block.

State units, frames, approximations and truncations where they affect the
answer. Use a component's declared parameters for the model inputs and
identify dressed quantities as results of the coupled system. Keep the
original model when you sweep, fit or reduce it.

## Show the result

Put the figure or compact numerical output below its calculation. Avoid
narrating every step with `print()`. A final machine-readable record can
collect parameters, observables and checks for automated verification, but it
does not replace the result shown to the reader.

Use familiar physical names. Say “bath model,” “pulse schedule,” or “parameter
sweep” instead of calling everything a “recipe.” `Bath.recipe` remains the
constructor argument for choosing a built-in bath model.

Give each page one title, shared by its heading, guide index and sidebar.
Titles should name the calculation rather than a quchip implementation concept.

## Plot the observable

Label axes with units and state which subsystem was traced out or which state
was prepared. Show the drive envelope when it explains the response. Use equal
axis scaling for IQ plots and a logarithmic scale when the range requires it.

Use the style in `docs/_static/quchip.mplstyle`: red, blue and charcoal, with
light and dark SVG variants and PDF downloads. Every committed figure must
come from the accompanying code. Inspect the rendered page in both themes.
Check labels, legends, figure size and collapsed code blocks.

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

Keep Jupytext Markdown and its executed notebook synchronized. Notebook
outputs stay in the notebook. Selected figures also go in `docs/images/`.
The guide includes the canonical Markdown under `docs/guides/`.

After execution, run `python tools/sync_example_outputs.py` to copy textual
outputs into the Markdown, then run it with `--check`. Inspect the figures and
run the focused example test and documentation build. The
{doc}`contributing guide <../contributing>` gives the execution and validation
commands.
