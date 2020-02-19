# brain2brain

### Conda Environment
To activate this environment, use

`$ conda activate brain2brain_env`

To deactivate an active environment, use

`$ conda deactivate`

*Note: `brain2brain_env.yml` contains only packages that were explicitly installed while `brain2brain_env_all.yml` contains all packages. `brain2brain_env.yml` contains only packages that are compatible across platforms.*

### Run model
Coming soon

### Experiments
Coming Soon

### Open Interactive Jupyter Lab on a GPU-enabled node.
First, queue a job. Make sure to edit `jupyter.sh` with the time you need.
`$ sbatch utils/jupyter.sh`

Second, look at the log file to see how to access the Jupyter Lab from your machine.
`$ cat jupyter...` (use TAB to autocomplete the filename)

In a different terminal:

`$ ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.princeton.edu` (where port, node, user, and cluster are specified in the jupyter.sh)

And then in a browser go to `localhost:8001`.

