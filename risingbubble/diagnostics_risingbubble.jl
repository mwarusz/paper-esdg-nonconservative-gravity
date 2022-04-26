
include("../common.jl")

include("risingbubble.jl")

using FileIO
using JLD2: @load
using NCDatasets
using DataStructures

function potential_temperature(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = constants(law).grav * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  p0 = 1e5
  T = p / (R_d * ρ)
  θ = T * (p0 / p)^(1 - 1 / constants(law).γ)
end

function diagnostics(law, q, qref, x⃗)
  θ = potential_temperature(law, q, x⃗)
  θ_ref = potential_temperature(law, qref, x⃗)
  δθ = θ - θ_ref
  SVector(δθ)
end

function calculate_diagnostics(data)
  diagnostic_data = Dict()

  law = data["law"]
  experiments = data["experiments"]
  for exp_key in keys(experiments)
    exp = experiments[exp_key]
    dg = exp.dg
    q = exp.q
    qref = exp.qref
    dη_timeseries = exp.dη_timeseries

    grid = dg.grid
    cell = referencecell(grid)
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = diagnostics.(Ref(law), q, qref, points(grid))
    diag_points, diag = interpolate_equidistant(diag, grid)
    diagnostic_data[(exp_key, N, KX)] = (; diag_points, diag, dη_timeseries)
  end

  diagnostic_data
end

function save_diagnostics(diagpath, diagnostic_data)
  for (exp, N, KX) in keys(diagnostic_data)
    data = diagnostic_data[(exp, N, KX)]
    x, z = components(data.diag_points)
    δθ = data.diag.:1
    dη_ts = data.dη_timeseries
    t = first.(dη_ts)
    δη = last.(dη_ts)

    ds = NCDataset(joinpath(diagpath, "data_$exp.nc"), "c")

    defDim(ds, "x", size(x, 1))
    defDim(ds, "z", size(z, 2))
    defDim(ds, "t", length(t))

    ncx = defVar(ds, "x", Float64, ("x",))
    ncz = defVar(ds, "z", Float64, ("z",))
    nct = defVar(ds, "t", Float64, ("t",))
    ncδθ = defVar(ds, "dtht", Float64, ("x", "z"))
    ncδη = defVar(ds, "deta", Float64, ("t",))

    ncx[:] = x[:, 1]
    ncz[:] = z[1, :]
    nct[:] = t
    ncδθ[:] = δθ
    ncδη[:] = δη

    close(ds)
  end
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  outpath = joinpath(outdir, "risingbubble", "jld2", "risingbubble.jld2")
  data = load(outpath)

  diagnostic_data = calculate_diagnostics(data)

  diagdir = length(ARGS) > 1 ? ARGS[2] : "diagnostics"
  diagpath = joinpath(diagdir, "risingbubble")
  mkpath(diagpath)

  save_diagnostics(diagpath, diagnostic_data)
end
