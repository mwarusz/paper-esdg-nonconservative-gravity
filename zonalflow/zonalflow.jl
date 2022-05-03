using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross

const _X = 125
const _a = 6.371229e6 / _X
const _p_0 = 1e5
const _grav = 9.80616
const _R_d = 287.0
const _T_0 = 287.0
const _u_0 = 20.0

struct ZonalFlow <: AbstractProblem end

longitude(x⃗) = @inbounds atan(x⃗[2], x⃗[1])
latitude(x⃗) = @inbounds asin(x⃗[3] / norm(x⃗))
function cartesian(v⃗, x⃗)
  u, v, w = v⃗
  r = norm(x⃗)
  λ = longitude(x⃗)
  φ = latitude(x⃗)

  uc = -sin(λ) * u - sin(φ) * cos(λ) * v + cos(φ) * cos(λ) * w
  vc = cos(λ) * u - sin(φ) * sin(λ) * v + cos(φ) * sin(λ) * w
  wc = cos(φ) * v + sin(φ) * w

  SVector(uc, vc, wc)
end

import Atum: boundarystate
function boundarystate(
  law::Union{LinearEulerGravityLaw, EulerGravityLaw},
  ::ZonalFlow,
  n⃗,
  q⁻,
  aux⁻,
  _,
)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, ::ZonalFlow, x⃗)
  FT = eltype(law)
  r = norm(x⃗)
  z = r - _a

  R_d = FT(_R_d)
  T_0 = FT(_T_0)
  p_0 = FT(_p_0)

  δ = constants(law).grav / (R_d * T_0)
  ρ_0 = p_0 / (T_0 * R_d)
  ρ_ref = ρ_0 * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_0

  SVector(ρ_ref, p_ref)
end

function zonalflow(law, x⃗, aux)
  FT = eltype(law)
  R_d = FT(_R_d)
  cv_d = R_d / (constants(law).γ - 1)
  grav = constants(law).grav
  p_0 = FT(_p_0)
  T_0 = FT(_T_0)
  u_0 = FT(_u_0)

  r = norm(x⃗)
  z = r - _a

  λ = longitude(x⃗)
  φ = latitude(x⃗)
  Φ = EulerGravity.geopotential(law, aux)

  f1 = z
  f2 = z / _a + z^2 / (2 * _a^2)
  shear = 1 + z / _a

  u_sphere = SVector{3, FT}(u_0 * shear * cos(φ), 0, 0)
  u_cart = cartesian(u_sphere, x⃗)

  prefac = u_0^2 / (R_d * T_0)
  fac1 = prefac * f2 * cos(φ)^2
  fac2 = prefac * sin(φ)^2 / 2
  fac3 = grav * f1 / (R_d * T_0)
  exparg = fac1 - fac2 - fac3
  p = p_0 * exp(exparg)

  ρ = p / (R_d * T_0)
  ρu⃗ = ρ * u_cart
  ρe = ρ * (cv_d * T_0 + u_cart' * u_cart / 2 + Φ)
  
  SVector(ρ, ρu⃗..., ρe)
end
