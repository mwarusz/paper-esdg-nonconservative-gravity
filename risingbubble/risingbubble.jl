using Atum
using Atum.EulerGravity

using LinearAlgebra: norm
using StaticArrays: SVector

const _L = 2e3
const _H = 2e3

struct RisingBubble <: AbstractProblem end

import Atum: boundarystate
function boundarystate(law::EulerGravityLaw, ::RisingBubble, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function gas_constant(law)
  FT = eltype(law)
  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d
end

function risingbubble(law, x⃗, add_perturbation = true)
  FT = eltype(law)
  x, z = x⃗

  Φ = constants(law).grav * z

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  θref = FT(300)
  p0 = FT(1e5)
  xc = FT(0)
  zc = FT(260)
  rc = FT(250)
  δθc = FT(1 / 2)

  r = sqrt((x - xc)^2 + (z - zc)^2)
  δθ = r <= rc ? δθc : zero(FT)

  θ = θref
  if add_perturbation
    θ += δθ
  end
  π_exner = 1 - constants(law).grav / (cp_d * θ) * z
  ρ = p0 / (R_d * θ) * π_exner^(cv_d / R_d)

  ρu = FT(0)
  ρv = FT(0)

  T = θ * π_exner
  ρe = ρ * (cv_d * T + Φ)

  SVector(ρ, ρu, ρv, ρe)
end
