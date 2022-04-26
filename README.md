# Experiments

- gravitywave
- risingbubble
- baroclinicwave

# Reproducibility pipeline

```
  julia --project=. experiment/run_experiment.jl [outdir="output"]
  julia --project=. experiment/diagnostics_experiment.jl [outdir="output"] [diagidir="diagnostics"]
  julia --project=. experiment/plot_experiment.jl [diagdir="diagnostics"] [plotdir="plots"]
```

# Example

- Reproduce gravitywave plots using provided reference diagnostics
```
julia --project=. gravitywave/plot_gravitywave.jl reference_diagnostics reference_plots
```

# Requirements

Latest [Atum](https://github.com/mwarusz/Atum.jl) 
