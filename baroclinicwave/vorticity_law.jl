module Vorticity
export VorticityLaw

import ..Atum
using StaticArrays
using StaticArrays: SUnitRange

struct VorticityLaw{FT, P} <: Atum.AbstractBalanceLaw{FT, 3, 3, (;)}
  problem::P
  function VorticityLaw{FT}(problem::P = Atum.DummyProblem()) where {FT, P}
    new{FT, P}(problem)
  end
end

function Atum.auxiliary(law::VorticityLaw, x⃗)
  FT = eltype(law)
  ρ = SVector(FT(0))
  ρu⃗ = zeros(SVector{3, FT})
  vcat(ρ, ρu⃗)
end

function density(law, aux)
  @inbounds aux[1]
end
function momentum(law, aux)
  @inbounds aux[SUnitRange(2, 2 + ndims(law) - 1)]
end

function Atum.flux(law::VorticityLaw, q, aux)
  ρ = density(law, aux)
  ρu⃗ = momentum(law, aux)
  u⃗ = ρu⃗ / ρ

  @inbounds begin
    @SMatrix [
      0 u⃗[3] -u⃗[2]
      -u⃗[3] 0 u⃗[1]
      u⃗[2] -u⃗[1] 0
    ]
  end
end

function Atum.wavespeed(law::VorticityLaw, n⃗, q, aux)
  ρ = density(law, aux)
  ρu⃗ = momentum(law, aux)
  u⃗ = ρu⃗ / ρ
  abs(n⃗' * u⃗)
end
end
