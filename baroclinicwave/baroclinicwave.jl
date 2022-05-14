using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross

const _X = 1
const _a = 6.371229e6 / _X
const _Ω = 7.29212e-5 * _X
const _p_0 = 1e5
const _grav = 9.80616
const _R_d = 287.0024093890231

struct BaroclinicWave <: AbstractProblem end

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
  ::BaroclinicWave,
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

import Atum: source!
function source!(
  law::EulerGravityLaw,
  ::BaroclinicWave,
  dq,
  q,
  aux,
  dim,
  directions,
)
  if dim ∈ directions
    FT = eltype(law)
    _, ix_ρu⃗, _ = EulerGravity.varsindices(law)
    @inbounds ρu⃗ = q[ix_ρu⃗]
    @inbounds dq[ix_ρu⃗] .-= cross(SVector{3, FT}(0, 0, 2_Ω), ρu⃗)
  end
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, ::BaroclinicWave, x⃗)
  FT = eltype(law)
  r = norm(x⃗)
  z = r - _a
  grav = constants(law).grav
  R_d = FT(_R_d)

  p_s = FT(_p_0)
  T_s = FT(290)
  T_min = FT(220)
  H_t = FT(8e3)

  H_sfc = R_d * T_s / grav
  z′ = z / H_t
  tanh_z′ = tanh(z′)

  ΔTv = T_s - T_min
  Tv = T_s - ΔTv * tanh_z′

  ΔTv′ = ΔTv / T_s
  p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
  p /= H_sfc * (1 - ΔTv′^2)
  p = p_s * exp(p)

  ρ = p / (R_d * Tv)

  SVector(ρ, p)
end

function baroclinicwave(law, x⃗, aux, add_perturbation = true)
  FT = eltype(law)
  R_d = FT(_R_d)
  cv_d = R_d / (constants(law).γ - 1)
  grav = constants(law).grav
  p_0 = FT(_p_0)
  Ω = FT(_Ω)
  a = FT(_a)
  Φ = EulerGravity.geopotential(law, aux)

  k = FT(3)
  T_E = FT(310)
  T_P = FT(240)
  T_0 = (T_E + T_P) / 2
  Γ = FT(0.005)
  A = 1 / Γ
  B = (T_0 - T_P) / T_0 / T_P
  C = (k + 2) / 2 * (T_E - T_P) / T_E / T_P
  b = 2
  H = R_d * T_0 / grav
  z_t = FT(15e3)
  λ_c = FT(π / 9)
  φ_c = FT(2 * π / 9)
  d_0 = FT(a / 6)
  V_p = FT(1)

  r = norm(x⃗)
  z = r - a
  λ = longitude(x⃗)
  φ = latitude(x⃗)

  γ = FT(1) # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

  # convenience functions for temperature and pressure
  τ_z_1 = exp(Γ * z / T_0)
  τ_z_2 = 1 - 2 * (z / b / H)^2
  τ_z_3 = exp(-(z / b / H)^2)
  τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
  τ_2 = C * τ_z_2 * τ_z_3
  τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
  τ_int_2 = C * z * τ_z_3
  I_T =
    (cos(φ) * (1 + γ * z / _a))^k -
    k / (k + 2) * (cos(φ) * (1 + γ * z / a))^(k + 2)

  # base state virtual temperature, pressure, specific humidity, density
  T = (τ_1 - τ_2 * I_T)^(-1)
  p = p_0 * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

  # base state velocity
  U =
    grav * k / a *
    τ_int_2 *
    T *
    ((cos(φ) * (1 + γ * z / a))^(k - 1) - (cos(φ) * (1 + γ * z / a))^(k + 1))
  u_ref =
    -Ω * (a + γ * z) * cos(φ) +
    sqrt((Ω * (a + γ * z) * cos(φ))^2 + (a + γ * z) * cos(φ) * U)
  v_ref = 0
  w_ref = 0

  # velocity perturbations
  F_z = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
  if z > z_t
    F_z = FT(0)
  end
  d = a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
  c3 = cos(π * d / 2 / d_0)^3
  s1 = sin(π * d / 2 / d_0)
  if 0 < d < d_0 && d != FT(a * π)
    u′ =
      -16 * V_p / 3 / sqrt(3) *
      F_z *
      c3 *
      s1 *
      (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) / sin(d / a)
    v′ =
      16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
      sin(d / a)
  else
    u′ = FT(0)
    v′ = FT(0)
  end
  w′ = FT(0)

  if add_perturbation
    u⃗_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
  else
    u⃗_sphere = SVector{3, FT}(u_ref, v_ref, w_ref)
  end
  u⃗ = cartesian(u⃗_sphere, x⃗)

  ρ = p / (R_d * T)
  ρu⃗ = ρ * u⃗
  ρe = ρ * (cv_d * T + u⃗' * u⃗ / 2 + Φ)

  SVector(ρ, ρu⃗..., ρe)
end
