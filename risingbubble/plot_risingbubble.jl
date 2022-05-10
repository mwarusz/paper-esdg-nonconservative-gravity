include("risingbubble.jl")
include("../common.jl")

using NCDatasets
using DataStructures

using PyPlot
using PGFPlotsX
using LaTeXStrings

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function rtb_contour_plot(plotpath, x, z, δθ)
  ioff()
  levels = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  fig = figure(figsize = (14, 12))
  ax = gca()
  xticks = range(-1000, 1000, length = 5)
  yticks = range(0, 2000, length = 5)
  ax.set_title("Potential temperature perturbation [K]")
  ax.set_xlim([xticks[1], xticks[end]])
  ax.set_ylim([yticks[1], yticks[end]])
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel(L"x" * " [m]")
  ax.set_ylabel(L"z" * " [m]")
  norm = matplotlib.colors.TwoSlopeNorm(
    vmin = levels[1],
    vcenter = 0,
    vmax = levels[end],
  )
  cset = ax.contourf(
    x',
    z',
    δθ',
    cmap = ColorMap("PuOr"),
    levels = levels,
    norm = norm,
  )
  ax.contour(x', z', δθ', levels = levels, colors = ("k",))
  ax.set_aspect(1)
  cbar = colorbar(cset)
  tight_layout()
  savefig(joinpath(plotpath, "rtb_tht_perturbation.pdf"))
end

function rtb_entropy_conservation_plot(plotpath, t_ec, dη_ec, t_mat, dη_mat)
  t_ec = t_ec[1:10:end]
  dη_ec = dη_ec[1:10:end]

  t_mat = t_mat[1:10:end]
  dη_mat = dη_mat[1:10:end]

  @pgf begin
    plot_ec = Plot({mark = "o", color = "red"}, Coordinates(t_ec, dη_ec))
    plot_matrix = Plot({mark = "x", color = "blue"}, Coordinates(t_mat, dη_mat))
    plot_crash = Plot({no_marks, dashed}, Coordinates([360, 360], [0, -2e-8]))
    legend = Legend("Entropy conservative flux", "Matrix dissipation flux")
    axis = Axis(
      {
        ylabel = L"(\eta - \eta_0) / |\eta_0|",
        xlabel = "time [s]",
        #ymode="log",
        legend_pos = "south west",
      },
      L"\node[] at (320,-1.0e-8) {vanilla DGSEM};",
      L"\node[] at (270,-1.2e-8) {breaks here};",
      plot_ec,
      plot_matrix,
      plot_crash,
      legend,
    )
    pgfsave(joinpath(plotpath, "rtb_entropy.pdf"), axis)
  end
end

let
  diagdir = length(ARGS) > 0 ? ARGS[1] : "diagnostics"
  diagpath = joinpath(diagdir, "risingbubble")

  plotdir = length(ARGS) > 1 ? ARGS[2] : "plots"
  plotpath = joinpath(plotdir, "risingbubble")
  mkpath(plotpath)

  Dataset(joinpath(diagpath, "data_hires_matrix.nc")) do ds
    x = ds["x"][:, :]
    z = ds["z"][:, :]
    δθ = ds["dtht"][:, :]
    rtb_contour_plot(plotpath, x, z, δθ)
  end

  entropy_cons_data = Dict()
  for numflux in ("ec", "matrix")
    Dataset(joinpath(diagpath, "data_lowres_$numflux.nc"), "r") do ds
      entropy_cons_data[numflux] = (t = ds["t"][:], dη = ds["deta"][:])
    end
  end
  rtb_entropy_conservation_plot(
    plotpath,
    entropy_cons_data["ec"]...,
    entropy_cons_data["matrix"]...,
  )
end
