julia = Base.julia_cmd()
base_dir = joinpath(@__DIR__, "..")
run_script = joinpath(@__DIR__, "run_gravitywave.jl")
diag_script = joinpath(@__DIR__, "diagnostics_gravitywave.jl")
plot_script = joinpath(@__DIR__, "plot_gravitywave.jl")

run(`$julia --project=$base_dir $run_script`)
run(`$julia --project=$base_dir $diag_script`)
run(`$julia --project=$base_dir $plot_script`)
