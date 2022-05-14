include("gravitywave.jl")
include("../common.jl")

using NCDatasets
using DataStructures

using PyPlot
using PGFPlotsX
using LaTeXStrings

const _scaling = 1e2 * _ΔT

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function gw_convergence_plot(plotpath, warp, polyorder, dx, T_error, w_error)
  dxs = dx[2, :]

  for (n, N) in enumerate(polyorder)
    @show N
    T_rates = log2.(T_error[n, 1:end-1] ./ T_error[n, 2:end])
    w_rates = log2.(w_error[n, 1:end-1] ./ w_error[n, 2:end])
    @show T_rates
    @show w_rates
  end

  @pgf begin
    plotsetup = {
      xlabel = "Δx [km]",
      grid = "major",
      xmode = "log",
      ymode = "log",
      xticklabel = "{\\pgfmathparse{exp(\\tick)/1000}\\pgfmathprintnumber[fixed,precision=3]{\\pgfmathresult}}",
      #xmax = 1,
      xtick = dxs,
      #ymin = 10. ^ -10 / 5,
      #ymax = 5,
      #ytick = 10. .^ -(0:2:10),
      legend_pos = "south east",
      group_style =
        {group_size = "2 by 2", vertical_sep = "2cm", horizontal_sep = "2cm"},
    }

    fig = GroupPlot(plotsetup)
    for s in ('T', 'w')
      unit = s == 'T' ? "K" : "m/s"
      ylabel = L"L_{2}" * " error of $s [$unit]"
      labels = []
      plots = []
      title = "Convergence of $s"
      for (n, N) in enumerate(polyorder)
        @show N
        dxs = dx[n, :]
        if warp == "nowarp"
          if N == 2
            Tcoeff = 2e-4
            wcoeff = 3e-4
          elseif N == 3
            Tcoeff = 2e-5
            wcoeff = 3e-5
          elseif N == 4
            Tcoeff = 5e-6
            wcoeff = 8e-6
          end
        else
          if N == 2
            Tcoeff = 5e-1
            wcoeff = 1e-2
          elseif N == 3
            Tcoeff = 9e-2
            wcoeff = 5e-3
          elseif N == 4
            Tcoeff = 1e-2
            wcoeff = 5e-4
          end
        end
        ordl = (dxs ./ dxs[1]) .^ (N + 1)
        if s === 'T'
          errs = T_error[n, :]
          ordl *= Tcoeff
        else
          errs = w_error[n, :]
          ordl *= wcoeff
        end
        coords = Coordinates(dxs, errs)
        ordc = Coordinates(dxs, ordl)
        plot = PlotInc({}, coords)
        plotc = Plot({dashed}, ordc)
        push!(plots, plot, plotc)
        #push!(labels, "N$N " * @sprintf("(%.2f)", rates[end]))
        push!(labels, "N$N")
        push!(labels, "order $(N+1)")
      end
      legend = Legend(labels)
      if s === 'T' && warp == "warp"
        push!(fig, {title = title, ylabel = ylabel, ymin=1e-10}, plots..., legend)
      else
        push!(fig, {title = title, ylabel = ylabel}, plots..., legend)
      end
    end
    savepath = joinpath(plotpath, "gw_convergence_$(warp).pdf")
    pgfsave(savepath, fig)
  end
end

function gw_add_contour_plot!(ax, KX, x, z, w, δT, w_exact, δT_exact)
  ll = 0.0036 * _scaling
  sl = 0.0006 * _scaling
  levels = vcat(-ll:sl:-sl, sl:sl:ll)
  xticks = range(0, 300, length = 7)
  yticks = range(0, 10, length = 5)

  dx = _L / KX

  ax[1].set_title(L"\Delta x" * " = $(dx) m\n\n w [m/s]")
  cset = ax[1].contourf(x', z', w', cmap = ColorMap("PuOr"), levels = levels)
  ax[1].contour(x', z', w_exact', levels = levels, colors = ("k",))

  ax[2].set_title("T perturbation [K]")
  norm = matplotlib.colors.TwoSlopeNorm(
    vmin = levels[1],
    vcenter = 0,
    vmax = levels[end],
  )
  cset = ax[2].contourf(
    x',
    z',
    δT',
    cmap = ColorMap("PuOr"),
    levels = levels,
    norm = norm,
  )
  ax[2].contour(x', z', δT_exact', levels = levels, colors = ("k",))

  for a in ax
    a.set_xlim([0, 300])
    a.set_xticks(xticks)
    a.set_yticks(yticks)
    a.set_ylim([0, 10])
    a.set_aspect(10)
  end

  ax[1].set_ylabel(L"z" * " [km]")
  ax[2].set_xlabel(L"x" * " [km]")

  cset, levels
end

function gw_contour_plot(plotpath, contour_data, KX)
  ioff()
  fig, ax = subplots(2, 1, figsize = (14, 10), sharex = "col", sharey = "row")

  cset, levels = gw_add_contour_plot!(ax, KX, contour_data[KX]...)

  tight_layout()
  cbar = colorbar(cset, ax = vec(ax), shrink = 1.0, ticks = levels)
  savefig(joinpath(plotpath, "gw_contour_$(KX).pdf"))
  close(fig)
end

function gw_compare_contour_plot(plotpath, contour_data, KX1, KX2)
  ioff()

  fig, ax = subplots(2, 2, figsize = (28, 10), sharex = "col", sharey = "row")

  cset, levels = gw_add_contour_plot!(ax[:, 1], KX1, contour_data[KX1]...)
  gw_add_contour_plot!(ax[:, 2], KX2, contour_data[KX2]...)

  tight_layout()
  cbar = colorbar(cset, ax = vec(ax), shrink = 1.0, ticks = levels)
  savefig(joinpath(plotpath, "gw_compare_contour_$(KX1)_vs_$(KX2).pdf"))
  close(fig)
end

function gw_add_line_plot!(fig, KX, x, z, w, δT, w_exact, δT_exact)
  k = findfirst(z .>= 5)
  w_k = w[:, k]
  w_exact_k = w_exact[:, k]

  ytick = [_scaling * (-3 + i) * 1e-3 for i = 0:7]
  xtick = [50 * i for i = 0:6]

  dx = _L / KX

  @pgf begin
    p1 = Plot({dashed}, Coordinates(x, w_k))
    p2 = Plot({}, Coordinates(x, w_exact_k))
    push!(
      fig,
      {
        xlabel = "x [km]",
        ylabel = "w [m/s]",
        ytick = ytick,
        xtick = xtick,
        title = L"\Delta x" * " = $dx m",
        width = "10cm",
        height = "5cm",
      },
      p1,
      p2,
    )
  end
end

function gw_line_plot(plotpath, contour_data, KX)
  fig = @pgf GroupPlot({
    group_style = {group_size = "1 by 1", vertical_sep = "1.5cm"},
    xmin = 0,
    xmax = 300,
  })
  gw_add_line_plot!(fig, KX, contour_data[KX]...)
  pgfsave(joinpath(plotpath, "gw_line_$(KX).pdf"), fig)
end

function gw_compare_line_plot(plotpath, contour_data, KX1, KX2)
  fig = @pgf GroupPlot({
    group_style = {group_size = "2 by 1", horizontal_sep = "1.5cm"},
    xmin = 0,
    xmax = 300,
  })

  gw_add_line_plot!(fig, KX1, contour_data[KX1]...)
  gw_add_line_plot!(fig, KX2, contour_data[KX2]...)

  pgfsave(joinpath(plotpath, "gw_compare_line_$(KX1)_vs_$(KX2).pdf"), fig)
end

let
  diagdir = length(ARGS) > 0 ? ARGS[1] : "diagnostics"
  diagpath = joinpath(diagdir, "gravitywave")

  plotdir = length(ARGS) > 1 ? ARGS[2] : "plots"
  plotpath = joinpath(plotdir, "gravitywave")
  mkpath(plotpath)

  KXs = (50, 100)
  contour_data = Dict()
  for KX in KXs
    Dataset(joinpath(diagpath, "contour_3_$(KX).nc"), "r") do ds
      contour_data[KX] = (
        x = ds["x"][:, 1],
        z = ds["z"][1, :],
        w = ds["w"][:, :],
        δT = ds["dT"][:, :],
        w_exact = ds["w exact"][:, :],
        δT_exact = ds["dT exact"][:, :],
      )
    end
  end
  for KX in KXs
    gw_contour_plot(plotpath, contour_data, KX)
    gw_line_plot(plotpath, contour_data, KX)
  end
  gw_compare_contour_plot(plotpath, contour_data, KXs...)
  gw_compare_line_plot(plotpath, contour_data, KXs...)

  for warp in ("nowarp", "warp")
    Dataset(joinpath(diagpath, "convergence_$(warp).nc"), "r") do ds
      polyorder = ds["polyorder"][:]
      dx = ds["dx"][:, :]
      T_error = ds["T error"][:, :]
      w_error = ds["w error"][:, :]
      gw_convergence_plot(plotpath, warp, polyorder, dx, T_error, w_error)
    end
  end
end
