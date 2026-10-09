# Installation

quchip requires Python 3.11 or newer. Install it in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install quchip
```

On Windows, activate with `.venv\Scripts\activate` instead.

## Optional integrations

QuTiP is the default simulation backend. Add integrations as needed:

```bash
python -m pip install 'quchip[dynamiqs]'  # JAX-native simulations
python -m pip install 'quchip[viz]'       # Chip and control graphs
python -m pip install 'quchip[scqubits]'  # scqubits interoperability
```

You can combine extras, for example `quchip[dynamiqs,viz]`. See
{doc}`backend and solver options <../guides/choosing-a-backend>` for numerical settings.

## Check the installation

```bash
python -c "import quchip; print(quchip.__version__)"
```

quchip is a 0.x project, so pin the version for reproducible
calculations, for example `quchip==0.4.0`.

Next, {doc}`declare your first chip <../guides/defining-and-inspecting-a-chip>`.
