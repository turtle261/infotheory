import Std
import ITE.Types

namespace ITE

open Std

/-- Convenience log base 2 for `Float`. -/
def log2 (x : Float) : Float :=
  Float.log x / Float.log 2.0

/-- Helper to build a map of oracle values. -/
def truthMap (xs : List (String × Float)) : Std.HashMap String Float :=
  xs.foldl (init := (HashMap.empty : HashMap String Float))
    (fun acc (k, v) => acc.insert k v)

/-- Shannon entropy of a finite categorical distribution. -/
def shannonEntropyCategorical (probs : Array Float) : Float :=
  probs.foldl (init := 0.0) (fun acc p =>
    if p ≤ 0.0 then acc else acc - p * log2 p)

/-- Bernoulli Shannon entropy. -/
def shannonEntropyBernoulli (p : Float) : Float :=
  if p == 0.0 || p == 1.0 then 0.0 else
    let q := 1.0 - p;
    -(p * log2 p + q * log2 q)

/-- KL divergence for a Bernoulli pair. -/
def klBernoulli (p q : Float) : Float :=
  let term a b := if a == 0.0 then 0.0 else a * log2 (a / b);
  term p q + term (1.0 - p) (1.0 - q)

/-- KL divergence for categorical distributions of equal length. -/
def klCategorical (p q : Array Float) : Except String Float := do
  if p.size ≠ q.size then
    throw "p and q must have identical support sizes"
  let mut acc := 0.0
  for i in List.range p.size do
    let pi := p.get! i
    let qi := q.get! i
    if pi == 0.0 then
      pure ()
    else if qi == 0.0 then
        throw "Support mismatch: qi = 0 where pi > 0"
      acc := acc + pi * log2 (pi / qi)
  return acc

/-- Mutual information for perfectly dependent variables given `H(X)`. -/
def miPerfectDependence (hX : Float) : Float := hX

/-- XOR mutual information for Ber(0.5) inputs. -/
def miXor : Float := 1.0

/-- Differential entropy of a Gaussian with variance `σ²`. -/
private def piF : Float := 3.141592653589793

def differentialEntropyGaussian (sigma : Float) : Float :=
  0.5 * log2 (2.0 * piF * Float.exp 1.0 * sigma * sigma)

/-- Differential entropy of a uniform on `[a,b]`. -/
def differentialEntropyUniform (a b : Float) : Float :=
  if b ≤ a then 0.0 else log2 (b - a)

/-- Differential entropy of an exponential with rate λ. -/
def differentialEntropyExponential (rate : Float) : Float :=
  log2 (Float.exp 1.0 / rate)

/-- Gaussian mutual information for a bivariate normal with correlation `ρ`. -/
def gaussianMutualInformation (rho sigmaX sigmaY : Float) : Float :=
  let _ignore := sigmaX -- kept for signature clarity
  let _ignore2 := sigmaY
  (-0.5) * log2 (1.0 - rho * rho)

/-- Oracle outcome: generated data plus truth values. -/
structure OracleOutcome where
  bundle : SampleBundle
  truths : Std.HashMap String Float
  deriving Repr

/-- Generative oracle record. -/
structure GenerativeOracle where
  name : String
  description : String
  generate : DataRegime → Nat → IO OracleOutcome

/-- Sample `n` values from `[0, alphabet)` uniformly. -/
private def sampleUniformNat (n alphabet : Nat) : IO (Array Float) := do
  let rec loop (k : Nat) (acc : Array Float) : IO (Array Float) := do
    if k = 0 then
      pure acc
    else
      let v ← IO.rand 0 (alphabet - 1)
      loop (k - 1) (acc.push (Float.ofNat v))
  loop n #[]

/-- Generate independent sources: `I(X;Y)=0`, `H(X,Y)=H(X)+H(Y)`. -/
def independentSourcesOracle : GenerativeOracle :=
  { name := "independent_sources"
  , description := "Independent uniform sources with known MI and joint entropy"
  , generate := fun regime nSamples => do
      let alphabet := match regime.alphabetSize with
        | .binary => 2
        | .small => 8
        | .medium => 64
        | .large => 1024
        | .continuous => 256  -- fallback discretization for sample generation
      let x ← sampleUniformNat nSamples alphabet
      let y ← sampleUniformNat nSamples alphabet
      let h := log2 (Float.ofNat alphabet)
      let truths := truthMap [
        ("I_XY", 0.0),
        ("H_X", h),
        ("H_Y", h),
        ("H_XY", 2.0 * h)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Deterministic function oracle: `Y = f(X)`, so `H(Y|X)=0`.

Note: `H(X|Y)=0` holds only when `f` is injective (e.g. `f = id`).
-/
def deterministicFunctionOracle (f : Float → Float) : GenerativeOracle :=
  { name := "deterministic_function"
  , description := "Deterministic mapping Y=f(X) with zero conditional entropy"
  , generate := fun regime nSamples => do
      let alphabet := match regime.alphabetSize with
        | .binary => 2
        | .small => 8
        | .medium => 64
        | .large => 1024
        | .continuous => 256
      let x ← sampleUniformNat nSamples alphabet
      let y := x.map f
      -- For a deterministic function, H(X|Y)=0 and I(X;Y)=H(Y).
      -- We do not attempt to empirically estimate H(Y) here; downstream estimators compute it.
      let truths := truthMap [
        ("H_X_given_Y", 0.0)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Binary symmetric channel oracle with known MI. -/
def noisyChannelOracle (flipProb : Float) : GenerativeOracle :=
  { name := "binary_symmetric_channel"
  , description := "Binary symmetric channel with analytic MI and conditional entropy"
  , generate := fun _regime nSamples => do
      let p0 := flipProb
      let p := if p0 < 0.0 then 0.0 else if p0 > 1.0 then 1.0 else p0
      let binaryEntropy (p : Float) : Float := shannonEntropyBernoulli p
      let hYgivenX := binaryEntropy p
      let hY := binaryEntropy 0.5
      let iXY := hY - hYgivenX

      let x ← sampleUniformNat nSamples 2
      let thresh := (p * 1000000.0).toUInt64.toNat
      let mut y : Array Float := #[]
      for xi in x do
        let r ← IO.rand 0 999999
        let flip := r < thresh
        y := y.push (if flip then 1.0 - xi else xi)

      let truths := truthMap [
        ("I_XY", iXY),
        ("H_Y_given_X", hYgivenX),
        ("H_Y", hY)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Bound oracle predicates that must hold for any valid estimator outputs. -/
structure BoundPredicate where
  name : String
  check : Std.HashMap String Float → Bool
  explanation : String

/-- Default bound predicates from the specification. -/
def boundPredicates : List BoundPredicate :=
  [ { name := "mi_nonnegative"
    , check := fun vals =>
        match vals.find? "I_XY" with
        | some v => v ≥ 0.0
        | none => true
    , explanation := "Mutual information must be non-negative"
    }
  , { name := "kl_nonnegative"
    , check := fun vals =>
        match vals.find? "D_KL" with
        | some v => v ≥ 0.0
        | none => true
    , explanation := "KL divergence must be non-negative"
    }
  , { name := "entropy_bound"
    , check := fun vals =>
        match vals.find? "H_X" with
        | some h =>
          match vals.find? "alphabet_size" with
          | some a => h ≤ log2 a
          | none => true
        | none => true
    , explanation := "Entropy cannot exceed log(|alphabet|)"
    }
  ]

end ITE
