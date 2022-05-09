include("baroclinicwave.jl")

using WriteVTK
using JLD2
using CUDA
using Adapt

function calc_p_and_vh(law, q, x⃗, aux)
  ρ, ρu⃗, ρe = EulerGravity.unpackstate(law, q)
  Φ = EulerGravity.geopotential(law, aux)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  u⃗ = ρu⃗ / ρ
  k⃗ = x⃗ / norm(x⃗)
  u⃗h = u⃗ - (k⃗' * u⃗) * u⃗
  vh = norm(u⃗h)
  SVector(p, vh)
end

function run(A, FT, law, linlaw, N, KH, KV; volume_form, outputvtk, outputjld2)
  Nq = N + 1

  ndays = 15

  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  modeltop = 30e3
  vr = range(FT(_a), stop = FT(_a + modeltop), length = KV + 1)
  grid = cubedspheregrid(cell, vr, KH)

  dg = DGSEM(; law, grid, volume_form, surface_numericalflux = MatrixFlux())

  dg_linear = DGSEM(;
    law = linlaw,
    grid,
    volume_form = WeakForm(),
    surface_numericalflux = RusanovFlux(),
    auxstate = dg.auxstate,
    directions = (3,),
  )

  cfl = FT(3)
  dz = min_node_distance(grid, dims = (3,))
  dt = cfl * dz / FT(330)
  @show dz, dt

  timeend = FT(ndays * 24 * 3600)

  numberofsteps = ceil(Int, timeend / dt)
  dt = timeend / numberofsteps

  q = fieldarray(undef, law, grid)
  q .= baroclinicwave.(Ref(law), points(grid), dg.auxstate, true)

  qref = fieldarray(undef, law, grid)
  qref .= baroclinicwave.(Ref(law), points(grid), dg.auxstate, false)

  if outputvtk
    vtkdir = joinpath("output", "baroclinicwave_hires", "vtk")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end
  count = 0
  do_output = function (step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      @show step, time
      filename = "KH_$(lpad(KH, 6, '0'))_KV_$(lpad(KV, 6, '0'))_step$(lpad(count, 6, '0'))"
      count += 1
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρw, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρw_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["δρw"] = vec(Array(P * (ρw - ρw_ref)))
      vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  qday = []
  save_days = function (step, time, q)
    if step % ceil(Int, timeend / 100 / dt) == 0
      @show step, time, 100 * time / timeend
    end
    if step % ceil(Int, timeend / ndays / dt) == 0
      push!(qday, adapt(Array, q))
    end
  end

  timeseries = NTuple{3, FT}[]
  tstime = 5 * 60
  save_vp = function (step, time, q)
    if step % ceil(Int, tstime / dt) == 0
      p, vh = components(calc_p_and_vh.(Ref(law), q, points(grid), dg.auxstate))
      min_p_surf = minimum(@view Array(p)[1:Nq^2, 1:KV:end])
      max_vh = maximum(Array(vh))
      @show time, min_p_surf, max_vh
      push!(timeseries, (time, min_p_surf, max_vh))
    end
  end

  callback = function (step, time, q)
    save_days(step, time, q)
    save_vp(step, time, q)
  end

  outputvtk && do_output(0, FT(0), q)
  callback(0, FT(0), q)
  odesolver = ARK23(
    dg,
    dg_linear,
    fieldarray(q),
    dt;
    split_rhs = false,
    paperversion = false,
  )
  solve!(q, timeend, odesolver; after_step = callback, adjust_final = false)
  outputvtk && vtk_save(pvd)

  dg = adapt(Array, dg)
  dg_linear = adapt(Array, dg_linear)
  (; dg, dg_linear, qday, timeseries)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  A = has_cuda_gpu() ? CuArray : Array
  FT = Float64

  law = EulerGravityLaw{FT, 3}(
    sphere = true,
    grav = _grav,
    problem = BaroclinicWave(),
  )
  linlaw = LinearEulerGravityLaw(law)

  outputvtk = false
  outputjld2 = true
  if outputjld2
    jld2dir = joinpath(outdir, "baroclinicwave", "jld2")
    mkpath(jld2dir)
  end

  experiments = Dict()

  N = length(ARGS) > 1 ? ARGS[2] : 3
  @assert(N in (3, 7))

  if N == 3
    KH = 30
    KV = 8
  else
    KH = 15
    KV = 4
  end

  if N >= 7
    volume_form = FluxDifferencingForm(EntropyConservativeFlux(), :naive)
  else
    volume_form = FluxDifferencingForm(EntropyConservativeFlux(), :per_dir)
  end
  experiments["$(N)_$(KH)_$(KV)"] =
    run(A, FT, law, linlaw, N, KH, KV; volume_form, outputvtk, outputjld2)

  if outputjld2
    @save(
      joinpath(jld2dir, "baroclinicwave_$(N).jld2"),
      law,
      linlaw,
      experiments
    )
  end
end
