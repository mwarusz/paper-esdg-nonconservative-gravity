using JLD2: load
using LinearAlgebra
using StaticArrays
using Polynomials
using NCDatasets
using DataStructures

include("gravitywave.jl")
include(joinpath("..", "common.jl"))

function diagnostics(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = constants(law).grav * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  ρ_ref, p_ref = referencestate(law, problem(law), x⃗)
  T_ref = p_ref / (R_d * ρ_ref)

  w = ρw / ρ
  δT = T - T_ref

  SVector(w, δT)
end

function compute_errors(dg, diag, diag_exact)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  err_w = weightednorm(dg, w .- w_exact) / sqrt(sum(dg.MJ))
  err_T = weightednorm(dg, δT .- δT_exact) / sqrt(sum(dg.MJ))

  err_w, err_T
end

function calculate_convergence(conv_exp, data)
  convergence_data = Dict()
  law = data["law"]
  experiments = data["experiments"]
  conv = experiments[conv_exp]

  for N in keys(conv)
    for exp in conv[N]
      dg = exp.dg
      q = exp.q
      qexact = exp.qexact

      grid = dg.grid
      cell = referencecell(grid)
      N = size(cell)[1] - 1
      KX = size(grid)[1]

      diag = diagnostics.(Ref(law), q, points(grid))
      diag_exact = diagnostics.(Ref(law), qexact, points(grid))

      dx = _L / KX
      err_w, err_T = compute_errors(dg, diag, diag_exact)
      @show dx, err_w, err_T

      if N in keys(convergence_data)
        push!(convergence_data[N].w_errors, err_w)
        push!(convergence_data[N].T_errors, err_T)
        push!(convergence_data[N].dxs, dx)
      else
        convergence_data[N] =
          (dxs = Float64[], w_errors = Float64[], T_errors = Float64[])
        push!(convergence_data[N].w_errors, err_w)
        push!(convergence_data[N].T_errors, err_T)
        push!(convergence_data[N].dxs, dx)
      end
    end
  end
  convergence_data
end

function calculate_diagnostics(data)
  diagnostic_data = Dict()

  law = data["law"]
  experiments = data["experiments"]

  for exp_key in keys(experiments)
    startswith(exp_key, "conv") && continue
    exp = experiments[exp_key]
    dg = exp.dg
    q = exp.q
    timeend = exp.timeend

    grid = dg.grid
    cell = referencecell(grid)
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = diagnostics.(Ref(law), q, points(grid))
    diag_points, diag = interpolate_equidistant(diag, grid)
    qexact = gravitywave.(Ref(law), diag_points, timeend)
    diag_exact = diagnostics.(Ref(law), qexact, diag_points)

    x, z = components(diag_points)
    # convert coordiantes to km
    x ./= 1e3
    z ./= 1e3

    diagnostic_data[(N, KX)] = (; diag_points, diag, diag_exact)
  end

  diagnostic_data
end

function save_diagnostics(diagpath, diagnostic_data)
  for (N, KX) in keys(diagnostic_data)
    data = diagnostic_data[(N, KX)]
    x, z = components(data.diag_points)

    w = data.diag.:1
    δT = data.diag.:2
    w_exact = data.diag_exact.:1
    δT_exact = data.diag_exact.:2

    ds = NCDataset(joinpath(diagpath, "contour_$(N)_$(KX).nc"), "c")
    defDim(ds, "i", size(x, 1))
    defDim(ds, "k", size(z, 2))

    ncx = defVar(ds, "x", Float64, ("i", "k"))
    ncz = defVar(ds, "z", Float64, ("i", "k"))
    ncw = defVar(ds, "w", Float64, ("i", "k"))
    ncδT = defVar(ds, "dT", Float64, ("i", "k"))
    ncw_exact = defVar(ds, "w exact", Float64, ("i", "k"))
    ncδT_exact = defVar(ds, "dT exact", Float64, ("i", "k"))

    ncx[:, :] = x
    ncz[:, :] = z
    ncw[:, :] = w
    ncδT[:, :] = δT
    ncw_exact[:, :] = w_exact
    ncδT_exact[:, :] = δT_exact

    close(ds)
  end
end

function save_convergence(diagpath, exp, convergence_data)
  ds = NCDataset(joinpath(diagpath, "convergence_$(exp).nc"), "c")

  orders = sort(collect(keys(convergence_data)))
  defDim(ds, "polyorder", length(orders))
  defDim(ds, "level", length(convergence_data[first(orders)].dxs))

  ncord = defVar(ds, "polyorder", Int, ("polyorder",))
  ncdxs = defVar(ds, "dx", Float64, ("polyorder", "level"))
  ncTerr = defVar(ds, "T error", Float64, ("polyorder", "level"))
  ncwerr = defVar(ds, "w error", Float64, ("polyorder", "level"))

  for (n, N) in enumerate(orders)
    ncord[n] = N
    ncdxs[n, :] = convergence_data[N].dxs
    ncTerr[n, :] = convergence_data[N].T_errors
    ncwerr[n, :] = convergence_data[N].w_errors
  end

  close(ds)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  outpath = joinpath(outdir, "gravitywave", "jld2", "gravitywave.jld2")
  data = load(outpath)
  diagnostic_data = calculate_diagnostics(data)
  convergence_data_nowarp = calculate_convergence("conv_nowarp", data)
  convergence_data_warp = calculate_convergence("conv_warp", data)

  diagdir = length(ARGS) > 1 ? ARGS[2] : "diagnostics"
  diagpath = joinpath(diagdir, "gravitywave")
  mkpath(diagpath)

  save_diagnostics(diagpath, diagnostic_data)
  save_convergence(diagpath, "nowarp", convergence_data_nowarp)
  save_convergence(diagpath, "warp", convergence_data_warp)
end
