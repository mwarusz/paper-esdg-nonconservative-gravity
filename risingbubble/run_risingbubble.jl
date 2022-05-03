include("risingbubble.jl")

using WriteVTK
using JLD2
using CUDA
using Adapt

function run(
  A,
  law,
  N,
  KX,
  KY,
  warp;
  volume_form = WeakForm(),
  surface_flux = RoeFlux(),
  vtkpath,
)
  outputvtk = !isnothing(vtkpath)
  FT = eltype(law)
  Nq = N + 1

  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(-_L / 2), stop = FT(_L / 2), length = KX + 1)
  vz = range(FT(0), stop = FT(_H), length = KY + 1)

  function meshwarp(x)
    x₁, x₂ = x
    x̃₁ = x₁ + sinpi((x₁ - first(vx)) / _L) * sinpi(2 * x₂ / _H) * _L / 5
    x̃₂ = x₂ - sinpi(2 * (x₁ - first(vx)) / _L) * sinpi(x₂ / _H) * _H / 5
    SVector(x̃₁, x̃₂)
  end

  grid = brickgrid(
    warp ? meshwarp : identity,
    cell,
    (vx, vz);
    periodic = (true, false),
  )

  dg = DGSEM(; law, grid, volume_form, surface_numericalflux = surface_flux)

  cfl = FT(4 // 10)
  dt =
    cfl * min_node_distance(grid) /
    EulerGravity.soundspeed(law, FT(1.16), FT(1e5))
  timeend = FT(1000)

  q = fieldarray(undef, law, grid)
  q .= risingbubble.(Ref(law), points(grid))
  qref = fieldarray(undef, law, grid)
  qref .= risingbubble.(Ref(law), points(grid), false)

  if outputvtk
    vtkpath = joinpath(vtkpath, "$surface_flux", "$N", "$(KX)x$(KY)")

    mkpath(vtkpath)
    pvd = paraview_collection(joinpath(vtkpath, "timesteps"))
  end

  dη_timeseries = NTuple{2, FT}[]
  η0 = entropyintegral(dg, q)
  do_output = function (step, time, q)
    if step % 100 == 0
      ηf = entropyintegral(dg, q)
      dη = (ηf - η0) / abs(η0)
      push!(dη_timeseries, (time, dη))
      @show step, time, dη
      flush(stdout)
    end
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkpath, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = RLSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)

  println("Starting")
  solve!(q, timeend, odesolver; after_step = do_output)
  println("Finished")
  outputvtk && vtk_save(pvd)

  dg = adapt(Array, dg)
  q = adapt(Array, q)
  qref = adapt(Array, qref)

  (; dg, q, qref, dη_timeseries)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  A = has_cuda_gpu() ? CuArray : Array
  FT = Float64

  outputjld = true
  outputvtk = false

  vtkpath = nothing
  if outputvtk
    vtkpath = joinpath(outdir, "risingbubble", "vtk")
  end

  if outputjld
    jldpath = joinpath(outdir, "risingbubble", "jld2")
    mkpath(jldpath)
  end

  volume_form =
    FluxDifferencingForm(EntropyConservativeFlux(), :per_dir_symmetric)
  pde_level_balance = volume_form isa WeakForm
  law = EulerGravityLaw{FT, 2}(; pde_level_balance, problem = RisingBubble())

  experiments = Dict()

  N = 4

  # entropy conservation, low res, warping
  warp = true
  KX = 10
  KY = 10
  experiments["lowres_ec"] = run(
    A,
    law,
    N,
    KX,
    KY,
    warp;
    volume_form,
    surface_flux = EntropyConservativeFlux(),
    vtkpath,
  )
  experiments["lowres_matrix"] = run(
    A,
    law,
    N,
    KX,
    KY,
    warp;
    volume_form,
    surface_flux = MatrixFlux(),
    vtkpath,
  )

  # plotting, high res, no warping
  warp = false
  KX = 40
  KY = 40
  experiments["hires_matrix"] = run(
    A,
    law,
    N,
    KX,
    KY,
    warp;
    volume_form,
    surface_flux = MatrixFlux(),
    vtkpath,
  )

  if outputjld
    @save(joinpath(jldpath, "risingbubble.jld2"), law, experiments)
  end
end
