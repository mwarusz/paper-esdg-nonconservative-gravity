include("gravitywave.jl")

using WriteVTK
using JLD2
using CUDA
using Adapt

function run(
  exp,
  A,
  law,
  N,
  KX,
  KY;
  volume_form = WeakForm(),
  surface_flux = RoeFlux(),
  warp,
  vtkpath,
)
  outputvtk = !isnothing(vtkpath)

  FT = eltype(law)
  Nq = N + 1

  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop = FT(_L), length = KX + 1)
  vz = range(FT(0), stop = FT(_H), length = KY + 1)

  function meshwarp(x)
    x₁, x₂ = x
    x̃₁ = x₁ + sinpi((x₁ - first(vx)) / _L) * sinpi(2 * x₂ / _H) * _L / 20
    x̃₂ = x₂ - sinpi(2 * (x₁ - first(vx)) / _L) * sinpi(x₂ / _H) * _H / 20
    SVector(x̃₁, x̃₂)
  end

  grid = brickgrid(warp ? meshwarp : identity, cell, (vx, vz); periodic = (true, false))

  dg = DGSEM(; law, grid, volume_form, surface_numericalflux = surface_flux)

  cfl = FT(1 // 10)
  dt =
    cfl * min_node_distance(grid) /
    EulerGravity.soundspeed(law, FT(1.16), FT(1e5))
  timeend = FT(30 * 60)

  q = fieldarray(undef, law, grid)
  q .= gravitywave.(Ref(law), points(grid), FT(0))
  qref = fieldarray(undef, law, grid)
  qref .= gravitywave.(Ref(law), points(grid), FT(0), false)

  if outputvtk
    vtkdir = joinpath(vtkpath, exp, "$N", "$(KX)x$(KY)")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function (step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
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

  odesolver = LSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step = do_output)
  outputvtk && vtk_save(pvd)

  qexact = gravitywave.(Ref(law), points(grid), timeend)
  errf = map(components(q), components(qexact)) do f, fexact
    sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
  end
  errf = norm(errf)

  dg = adapt(Array, dg)
  q = adapt(Array, q)
  qexact = adapt(Array, qexact)

  (; dg, q, qexact, errf, timeend)
end

let
  outdir = length(ARGS) > 0 ? ARGS[1] : "output"
  A = has_cuda_gpu() ? CuArray : Array
  FT = Float64

  volume_form =
    FluxDifferencingForm(EntropyConservativeFlux(), :per_dir_symmetric)
  surface_flux = MatrixFlux()
  pde_level_balance = volume_form isa WeakForm
  law = EulerGravityLaw{FT, 2}(; pde_level_balance, problem = GravityWave())

  outputvtk = false
  outputjld = true

  vtkpath = nothing
  if outputvtk
    vtkpath = joinpath(outdir, "gravitywave", "vtk")
    mkpath(vtkpath)
  end
  if outputjld
    jldpath = joinpath(outdir, "gravitywave", "jld2")

    mkpath(jldpath)
  end

  experiments = Dict()

  N = 3

  KX = 50
  KY = 5
  exp = "dx6"
  experiments[exp] =
    run(exp, A, law, N, KX, KY; volume_form, surface_flux, vtkpath, warp=false)

  KX = 100
  KY = 10
  exp = "dx3"
  experiments[exp] =
    run(exp, A, law, N, KX, KY; volume_form, surface_flux, vtkpath, warp=false)

  # convergence without warping
  exp = "conv_nowarp"
  experiments["conv_nowarp"] = Dict()
  KX_base = 30
  KY_base = 3
  polyorders = 2:4
  nlevels = 4
  for N in polyorders
    experiments[exp][N] = ntuple(nlevels) do l
      KX = KX_base * 2^(l - 1)
      KY = KY_base * 2^(l - 1)
      @show l, KX, KY
      run(exp, A, law, N, KX, KY; volume_form, surface_flux, vtkpath, warp=false)
    end
  end
  
  # convergence with warping
  nlevels = 5
  exp = "conv_warp"
  experiments[exp] = Dict()
  for N in polyorders
    experiments[exp][N] = ntuple(nlevels) do l
      KX = KX_base * 2^(l - 1)
      KY = KY_base * 2^(l - 1)
      @show l, KX, KY
      run(exp, A, law, N, KX, KY; volume_form, surface_flux, vtkpath, warp=true)
    end
  end

  if outputjld
    @save(joinpath(jldpath, "gravitywave.jld2"), law, experiments)
  end

  for N in polyorders
    errors = zeros(FT, nlevels)
    nlevels = 4
    for l = 1:nlevels
      errors[l] = experiments["conv_nowarp"][N][l].errf
    end
    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @show N
      @show errors
      @show rates
    end
    
    nlevels = 5
    for l = 1:nlevels
      errors[l] = experiments["conv_warp"][N][l].errf
    end
    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @show N
      @show errors
      @show rates
    end
  end
end
