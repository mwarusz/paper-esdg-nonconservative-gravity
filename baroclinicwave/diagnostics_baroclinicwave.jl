
include("baroclinicwave.jl")
include("vorticity_law.jl")

using JLD2: load
using .Vorticity: VorticityLaw
using LinearAlgebra
using StaticArrays
using Polynomials
using NCDatasets
using DataStructures

function diagnostics(law, q, aux, ω⃗, x⃗)
  FT = eltype(law)
  ρ, ρu⃗, ρe = EulerGravity.unpackstate(law, q)
  Φ = EulerGravity.geopotential(law, aux)

  R_d = FT(_R_d)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  k⃗ = x⃗ / norm(x⃗)

  SVector(p / 1e2, T, k⃗' * ω⃗ / 1e-5)
end

function spherical_coordinates(x⃗)
  r = norm(x⃗)
  x, y, z = x⃗

  lat = 180 / π * asin(z / r)
  lon = 180 / π * atan(y, x)

  SVector(lat, lon)
end

function interpolate_horz_sphere(data, grid)
  cell = referencecell(grid)
  dim = ndims(cell)

  ξsrc = vec.(cell.points_1d)
  Nq = size(cell)

  #Nqi = ntuple(d -> 4 * Nq[d], 2)
  Nqi = ntuple(d -> 1 * Nq[d], 2)
  dξi = 2 ./ Nqi
  ξdst = ntuple(d -> [-1 + (j - 1 / 2) * dξi[d] for j = 1:Nqi[d]], 2)
  I = kron(
    LinearAlgebra.I(Nq[3]),
    (ntuple(d -> spectralinterpolation(ξsrc[d], ξdst[d]), 2))...,
  )

  points_i = I * points(grid)
  data_i = I * data

  Nqi[1], points_i, data_i
end

function interpolate_to_pressure_surface(p, T, ωk, N, Nqi, KH, KV; psurf = 850)
  FT = eltype(p)
  Nq = N + 1

  T850 = Array{FT}(undef, Nqi^2, 6 * KH^2)
  T850 .= NaN

  ωk850 = Array{FT}(undef, Nqi^2, 6 * KH^2)
  ωk850 .= NaN

  for eh = 1:6*KH^2
    for i = 1:Nqi
      for j = 1:Nqi
        for ev = 1:KV
          e = ev + (eh - 1) * KV
          pe = MVector{Nq, FT}(undef)
          Te = MVector{Nq, FT}(undef)
          ωke = MVector{Nq, FT}(undef)
          for k = 1:Nq
            ijk = i + Nqi * (j - 1 + Nqi * (k - 1))
            pe[k] = p[ijk, e]
            Te[k] = T[ijk, e]
            ωke[k] = ωk[ijk, e]
          end
          pmin, pmax = extrema(pe)
          ξ, _ = legendregausslobatto(FT, Nq)
          V = vander(Polynomial{FT}, ξ, N)
          pl = Polynomial(V \ pe)
          ptest = pl(FT(-1))

          if pmin <= psurf <= pmax
            r = filter(isreal, roots(pl - psurf))
            r = real.(r)
            r = filter(x -> -1 <= x <= 1, r)
            @assert length(r) == 1
            I = spectralinterpolation(ξ, r)
            Tr = I * Te
            ωkr = I * ωke
            ij = i + Nqi * (j - 1)
            T850[ij, eh] = Tr[1]
            ωk850[ij, eh] = ωkr[1]
          end
        end
      end
    end
  end
  T850, ωk850
end

function compute_vorticity(q, grid)
  ρ, ρu, ρv, ρw, ρe = components(q)
  FT = eltype(ρ)

  law = VorticityLaw{FT}()

  vort_dg = DGSEM(;
    law,
    grid,
    volume_form = WeakForm(),
    surface_numericalflux = RusanovFlux(),
  )

  vort_dg.auxstate.:1 .= ρ
  vort_dg.auxstate.:2 .= ρu
  vort_dg.auxstate.:3 .= ρv
  vort_dg.auxstate.:4 .= ρw

  ω = similar(points(grid))
  vort_dg(ω, ω, nothing; increment = false)
  ω
end

function calculate_diagnostics(data)
  diagnostic_data = Dict()

  law = data["law"]
  experiments = data["experiments"]

  for exp_key in keys(experiments)
    exp = experiments[exp_key]
    dg = exp.dg
    qday = exp.qday

    grid = dg.grid
    cell = referencecell(grid)
    N = size(cell)[1] - 1
    KV = size(grid)[1]
    KH = size(grid)[2]

    day = 1
    for q in qday
      @show day
      vort = compute_vorticity(q, grid)
      diag = diagnostics.(Ref(law), q, dg.auxstate, vort, points(grid))
      p, T, ωk = components(diag)

      Nqi, points_i, diag = interpolate_horz_sphere(diag, grid)
      diag = vcat.(diag, spherical_coordinates.(points_i))

      p, T, ωk, lat, lon = components(diag)
      @show day, extrema(p), extrema(ωk), extrema(lat), extrema(lon)

      T850, ωk850 = interpolate_to_pressure_surface(p, T, ωk, N, Nqi, KH, KV)
      @show extrema(T850), extrema(ωk850)

      psurf = @view p[1:Nqi^2, 1:KV:end][:]
      lat = @view lat[1:Nqi^2, 1:KV:end][:]
      lon = @view lon[1:Nqi^2, 1:KV:end][:]
      T850 = @view T850[:]
      ωk850 = @view ωk850[:]

      diagnostic_data[day] = (; lat, lon, psurf, T850, ωk850)
      day += 1
    end

    diagnostic_data["tseries"] = exp.timeseries
  end

  diagnostic_data
end

function save_data(diagpath, diagnostic_data)
  ds = NCDataset(joinpath(diagpath, "baroclinicwave_3.nc"), "c")
  ndays = length(keys(diagnostic_data))
  npoints = length(diagnostic_data[first(keys(diagnostic_data))].lat)

  tseries = diagnostic_data["tseries"]

  defDim(ds, "point", npoints)
  defDim(ds, "day", ndays)
  defDim(ds, "t", length(tseries))

  lat = defVar(ds, "lat", Float64, ("point",))
  lon = defVar(ds, "lon", Float64, ("point",))
  psurf = defVar(ds, "psurf", Float64, ("day", "point"))
  T850 = defVar(ds, "T850", Float64, ("day", "point"))
  vort850 = defVar(ds, "vort850", Float64, ("day", "point"))

  t = defVar(ds, "t", Float64, ("t",))
  min_psurf = defVar(ds, "min psurf", Float64, ("t",))
  max_vh = defVar(ds, "max vh", Float64, ("t",))

  lat[:] = diagnostic_data[1].lat
  lon[:] = diagnostic_data[1].lon

  t[:] = first.(tseries)
  min_psurf[:] = map(x -> x[2], tseries)
  max_vh[:] = last.(tseries)

  for day in keys(diagnostic_data)
    day isa Int || continue
    data = diagnostic_data[day]
    psurf[day, :] = data.psurf
    T850[day, :] = data.T850
    vort850[day, :] = data.ωk850
  end

  close(ds)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  outpath = joinpath(outdir, "baroclinicwave", "jld2", "baroclinicwave_3.jld2")
  data = load(outpath)
  diagnostic_data = calculate_diagnostics(data)

  diagdir = length(ARGS) > 1 ? ARGS[2] : "diagnostics"
  diagpath = joinpath(diagdir, "baroclinicwave")
  mkpath(diagpath)
  save_data(diagpath, diagnostic_data)
end
