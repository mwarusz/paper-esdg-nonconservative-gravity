
using NCDatasets
using DataStructures

using PGFPlotsX
using PyPlot
function rcParams!(rcParams)
  rcParams["font.size"] = 20
  rcParams["xtick.labelsize"] = 20
  rcParams["ytick.labelsize"] = 20
  rcParams["legend.fontsize"] = 20
  rcParams["figure.titlesize"] = 32
  rcParams["axes.titlepad"] = 10
  rcParams["axes.labelpad"] = 10
  rcParams["backend"] = "pdf"
end

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

const lonshift = 60

function bw_compare_days(plotpath, lat, lon, day1, day2, day1data, day2data)
  ioff()
  fig, axs = subplots(3, 2, figsize = (27, 20))

  plevels1 = 10
  pnorm1 = nothing
  if day1 == 8
    plevels1 = vcat([955], [960 + 5i for i = 0:11], [1020])
    pnorm1 =
      matplotlib.colors.TwoSlopeNorm(vmin = 955, vcenter = 990, vmax = 1025)
  end

  plevels2 = 10
  pnorm2 = nothing
  if day2 == 10
    plevels2 = vcat([920], [930 + 10i for i = 0:9], [1030])
    pnorm2 =
      matplotlib.colors.TwoSlopeNorm(vmin = 920, vcenter = 980, vmax = 1040)
  end

  bw_add_contour_plot!(axs[:, 1], day1, lon, lat, day1data..., plevels1, pnorm1)
  bw_add_contour_plot!(axs[:, 2], day2, lon, lat, day2data..., plevels2, pnorm2)

  plt.subplots_adjust(wspace = 0.05)
  savefig(joinpath(plotpath, "bw_panel.pdf"))
  close(fig)
end

function bw_add_contour_plot!(
  axs,
  day,
  lon,
  lat,
  psurf,
  T850,
  ωk850,
  plevels,
  pnorm,
)
  dayi = day == 8 ? 1 : 2
  lon = @. mod(lon + 180 - lonshift, 360) - 180
  mask = (lat .> 0) .& (lon .> -lonshift)
  lon = lon[mask]
  lat = lat[mask]
  T850 = T850[mask]
  ωk850 = ωk850[mask]
  psurf = psurf[mask]

  cmap = ColorMap("nipy_spectral").copy()
  shrinkcb = 0.7

  axs[1].tricontour(lon, lat, psurf; levels = plevels, colors = ("k",))
  cset = axs[1].tricontourf(
    lon,
    lat,
    psurf;
    levels = plevels,
    cmap,
    norm = pnorm,
    extend = "neither",
  )
  axs[1].set_title("Surface pressure", loc = "left")
  axs[1].set_title("Day $day", loc = "center")
  axs[1].set_title("hPa", loc = "right")
  cbar = colorbar(
    cset,
    orientation = "horizontal",
    ax = axs[1],
    ticks = plevels isa Int ? nothing : plevels[1+dayi:2:end-1],
    shrink = shrinkcb,
  )

  levels = vcat([220], [230 + 10i for i = 0:7], [310])
  norm = matplotlib.colors.TwoSlopeNorm(vmin = 220, vcenter = 270, vmax = 320)
  axs[2].tricontour(lon, lat, T850; levels, colors = ("k",))
  cset =
    axs[2].tricontourf(lon, lat, T850; levels, cmap, norm, extend = "neither")
  axs[2].set_title("850 hPa Temperature", loc = "left")
  axs[2].set_title("Day $day", loc = "center")
  axs[2].set_title("K", loc = "right")
  cbar = colorbar(
    cset,
    orientation = "horizontal",
    ax = axs[2],
    ticks = levels[2:end-1],
    shrink = shrinkcb,
  )

  cmap = ColorMap("seismic").copy()
  levels = vcat([-10], [-5 + 5i for i = 0:6], [30])
  norm = matplotlib.colors.TwoSlopeNorm(vmin = -10, vcenter = 0, vmax = 30)
  axs[3].tricontour(lon, lat, ωk850; levels, colors = ("k",))
  cset =
    axs[3].tricontourf(lon, lat, ωk850; levels, cmap, norm, extend = "neither")
  axs[3].set_title("850 hPa Vorticity", loc = "left")
  axs[3].set_title("Day $day", loc = "center")
  axs[3].set_title("1e-5/s", loc = "right")
  cbar = colorbar(
    cset,
    orientation = "horizontal",
    ax = axs[3],
    ticks = levels[2:end-1],
    shrink = shrinkcb,
  )

  xticks = [-60, -30, 0, 30, 60, 90, 120, 150, 180]
  xticklabels =
    ["0", "30E", "60E", "90E", "120E", "150E", "180", "150W", "120W"]
  yticks = [0, 30, 60, 90]
  yticklabels = ["0", "30N", "60N", "90N"]
  for ax in axs[:]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim([xticks[1], xticks[end]])
    ax.set_ylim([yticks[1], yticks[end]])
    ax.set_aspect(1)
  end
end

function bw_timeseries(plotpath, t, pmin, vmax)
  t /= (24 * 3600)
  pmin /= 100

  @pgf begin
    xtick = 0:3:12
    ptick = 870:30:990

    fig = @pgf GroupPlot({
      group_style = {group_size = "2 by 1", horizontal_sep = "1.5cm"},
      xtick = xtick,
      xmin = 0,
      xmax = 12,
      xlabel = "Day",
    })
    ppmin = Plot({}, Coordinates(t, pmin))
    push!(
      fig,
      {ytick = ptick, ylabel = "Minimum Surface Pressure [hPa]"},
      ppmin,
    )

    pvmax = Plot({}, Coordinates(t, vmax))
    push!(
      fig,
      {ytick = ptick, ylabel = "Maximum Horizontal Wind Speed [m/s]"},
      pvmax,
    )

    pgfsave(joinpath(plotpath, "bw_tseries.pdf"), fig)
  end
end

let
  diagdir = length(ARGS) > 0 ? ARGS[1] : "diagnostics"
  diagpath = joinpath(diagdir, "baroclinicwave")

  plotdir = length(ARGS) > 1 ? ARGS[2] : "plots"
  plotpath = joinpath(plotdir, "baroclinicwave")
  mkpath(plotpath)

  compare_days = (8, 10)
  tseries_data = Dict()

  Dataset(joinpath(diagpath, "baroclinicwave_3.nc"), "r") do ds
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    tseries_data[3] =
      (t = ds["t"][:], min_psurf = ds["min psurf"][:], max_vh = ds["max vh"][:])

    daydata = ntuple(2) do i
      day = compare_days[i]
      psurf = ds["psurf"][day+1, :]
      T850 = ds["T850"][day+1, :]
      ωk850 = ds["vort850"][day+1, :]
      (psurf, T850, ωk850)
    end
    bw_compare_days(plotpath, lat, lon, compare_days..., daydata...)
  end

  bw_timeseries(plotpath, tseries_data[3]...)
end
