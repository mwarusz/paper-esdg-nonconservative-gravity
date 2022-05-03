include("zonalflow.jl")

using WriteVTK
using JLD2
using CUDA
using Adapt

function run(A, FT, law, N, KH, KV; volume_form, outputvtk, outputjld2)
  Nq = N + 1

  ndays = 1

  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  modeltop = 10e3
  vr = range(FT(_a), stop = FT(_a + modeltop), length = KV + 1)
  grid = cubedspheregrid(cell, vr, KH)

  #dg = DGSEM(; law, grid, volume_form, surface_numericalflux = EntropyConservativeFlux())
  dg = DGSEM(; law, grid, volume_form, surface_numericalflux = MatrixFlux())

  cfl = FT(0.2)
  dx = min_node_distance(grid)
  dt = cfl * dx / FT(330)

  timeend = FT(ndays * 24 * 3600 / _X)

  q = fieldarray(undef, law, grid)
  q .= zonalflow.(Ref(law), points(grid), dg.auxstate)

  qref = fieldarray(undef, law, grid)
  qref .= zonalflow.(Ref(law), points(grid), dg.auxstate)

  #if outputvtk
  #  vtkdir = joinpath("output", "baroclinicwave_hires", "vtk")
  #  mkpath(vtkdir)
  #  pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  #end
  #count = 0
  #do_output = function (step, time, q)
  #  if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
  #    @show step, time
  #    filename = "KH_$(lpad(KH, 6, '0'))_KV_$(lpad(KV, 6, '0'))_step$(lpad(count, 6, '0'))"
  #    count += 1
  #    vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
  #    P = Bennu.toequallyspaced(cell)
  #    ρ, ρu, ρv, ρw, ρe = components(q)
  #    ρ_ref, ρu_ref, ρv_ref, ρw_ref, ρe_ref = components(qref)
  #    vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
  #    vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
  #    vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
  #    vtkfile["δρw"] = vec(Array(P * (ρw - ρw_ref)))
  #    vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref)))
  #    vtk_save(vtkfile)
  #    pvd[time] = vtkfile
  #  end
  #end

  dη_timeseries = NTuple{2, FT}[]
  η0 = entropyintegral(dg, q)
  callback = function (step, time, q)
    if step % 100 == 0
      ηf = entropyintegral(dg, q)
      dη = (ηf - η0) / abs(η0)
      push!(dη_timeseries, (time, dη))
      dq = weightednorm(dg, q .- qref)
      @show step, time, dη, dq
      flush(stdout)
    end
  end

  odesolver = LSRK54(dg, q, dt)
  
  #outputvtk && do_output(0, FT(0), q)
  callback(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step = callback)
  #outputvtk && vtk_save(pvd)

  errf = weightednorm(dg, q .- qref)
  dg = adapt(Array, dg)
  (; dg, errf, dη_timeseries)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  A = has_cuda_gpu() ? CuArray : Array
  FT = Float64
  volume_form = FluxDifferencingForm(EntropyConservativeFlux(), :per_dir)

  law = EulerGravityLaw{FT, 3}(
    sphere = true,
    grav = _grav,
    problem = ZonalFlow(),
  )

  outputvtk = false
  outputjld2 = false
  if outputjld2
    jld2dir = joinpath(outdir, "zonalflow", "jld2")
    mkpath(jld2dir)
  end

  experiments = Dict()
  experiments["conv"] = Dict()

  polyorders = 2:4
  nlevels = 3
  KH_base = 4
  KV_base = 4

  for N in polyorders
    experiments["conv"][N] = ntuple(nlevels) do l
      KH = KH_base * 2^(l - 1)
      KV = KV_base * 2^(l - 1)
      res = run(A, FT, law, N, KH, KV; volume_form, outputvtk, outputjld2)
      @show l, res.errf
      res
    end
  end

  for N in polyorders
    errors = zeros(FT, nlevels)
    for l = 1:nlevels
      errors[l] = experiments["conv"][N][l].errf
    end
    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @show N
      @show errors
      @show rates
    end
  end
end
