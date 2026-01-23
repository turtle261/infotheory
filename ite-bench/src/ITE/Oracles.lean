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

/-- Deterministic linear congruential generator for reproducible tests. -/
structure Lcg where
  state : UInt64
  deriving Repr

def Lcg.next (g : Lcg) : Lcg :=
  -- LCG parameters from Numerical Recipes (full-period for 2^64)
  { state := 6364136223846793005 * g.state + 1 }

def Lcg.nextNat (g : Lcg) (bound : Nat) : (Nat × Lcg) :=
  if bound = 0 then (0, g)
  else
    let g' := g.next
    let hi := (g'.state >>> 32).toNat
    let v := hi % bound
    (v, g')

def Lcg.nextFloat01 (g : Lcg) : (Float × Lcg) :=
  let (n, g') := g.nextNat 1000000
  let v := (Float.ofNat n) / 1000000.0
  (v, g')

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

def normalizeProbs (probs : Array Float) : Array Float :=
  let s := probs.foldl (init := 0.0) (· + ·)
  if s == 0.0 then probs else probs.map (· / s)

def normalizeJoint (joint : Array (Array Float)) : Array (Array Float) :=
  let total := joint.foldl (init := 0.0) (fun acc row =>
    acc + row.foldl (init := 0.0) (· + ·))
  if total == 0.0 then joint else joint.map (fun row => row.map (· / total))

def tvdCategorical (p q : Array Float) : Except String Float := do
  if p.size ≠ q.size then
    throw "p and q must have identical support sizes"
  let mut acc := 0.0
  for i in List.range p.size do
    acc := acc + Float.abs (p.get! i - q.get! i)
  return 0.5 * acc

def jsCategorical (p q : Array Float) : Except String Float := do
  if p.size ≠ q.size then
    throw "p and q must have identical support sizes"
  let mut m : Array Float := Array.empty
  for i in List.range p.size do
    m := m.push (0.5 * (p.get! i + q.get! i))
  let kl1 ← klCategorical p m
  let kl2 ← klCategorical q m
  return 0.5 * (kl1 + kl2)

def jointEntropyCategorical (joint : Array (Array Float)) : Float :=
  joint.foldl (init := 0.0) (fun acc row =>
    acc + row.foldl (init := 0.0) (fun acc2 p =>
      if p ≤ 0.0 then acc2 else acc2 - p * log2 p))

def jointToMarginals (joint : Array (Array Float)) : (Array Float × Array Float) :=
  Id.run do
    let rows := joint.size
    let cols := if rows == 0 then 0 else (joint.get! 0).size
    let mut px := Array.mkArray rows 0.0
    let mut py := Array.mkArray cols 0.0
    for i in List.range rows do
      let row := joint.get! i
      for j in List.range cols do
        let v := row.get! j
        px := px.set! i (px.get! i + v)
        py := py.set! j (py.get! j + v)
    return (px, py)

def sampleUniformNatDet (n alphabet : Nat) (g : Lcg) : (Array Float × Lcg) :=
  let rec loop (k : Nat) (acc : Array Float) (g : Lcg) : (Array Float × Lcg) :=
    if k = 0 then (acc, g)
    else
      let (v, g') := g.nextNat alphabet
      loop (k - 1) (acc.push (Float.ofNat v)) g'
  loop n #[] g

def sampleCategoricalDet (n : Nat) (probs : Array Float) (g : Lcg) : (Array Float × Lcg) :=
  let probs := normalizeProbs probs
  let rec findIdx (xs : List Float) (u : Float) (i : Nat) : Nat :=
    match xs with
    | [] => if probs.size == 0 then 0 else probs.size - 1
    | x :: xs =>
        if u ≤ x then i else findIdx xs u (i + 1)
  let cdf : Array Float := Id.run do
    let mut acc := 0.0
    let mut out := #[]
    for p in probs do
      acc := acc + p
      out := out.push acc
    return out
  let choose (u : Float) : Nat :=
    findIdx cdf.toList u 0
  let rec loop (k : Nat) (acc : Array Float) (g : Lcg) : (Array Float × Lcg) :=
    if k = 0 then (acc, g)
    else
      let (u, g') := Lcg.nextFloat01 g
      let idx := choose u
      loop (k - 1) (acc.push (Float.ofNat idx)) g'
  loop n #[] g

def sampleJointCategoricalDet (n : Nat) (joint : Array (Array Float)) (g : Lcg) :
    (Array Float × Array Float × Lcg) :=
  let joint := normalizeJoint joint
  let rows := joint.size
  let cols := if rows == 0 then 0 else (joint.get! 0).size
  let flat : Array Float :=
    (List.range rows).foldl (init := #[]) (fun acc i =>
      (List.range cols).foldl (init := acc) (fun acc2 j =>
        acc2.push ((joint.get! i).get! j)))
  let rec findIdx (xs : List Float) (u : Float) (i : Nat) : Nat :=
    match xs with
    | [] => if flat.size == 0 then 0 else flat.size - 1
    | x :: xs =>
        if u ≤ x then i else findIdx xs u (i + 1)
  let cdf : Array Float := Id.run do
    let mut acc := 0.0
    let mut out := #[]
    for p in flat do
      acc := acc + p
      out := out.push acc
    return out
  let choose (u : Float) : Nat :=
    findIdx cdf.toList u 0
  let rec loop (k : Nat) (xs ys : Array Float) (g : Lcg) : (Array Float × Array Float × Lcg) :=
    if k = 0 then (xs, ys, g)
    else
      let (u, g') := Lcg.nextFloat01 g
      let idx := choose u
      let i := idx / cols
      let j := idx % cols
      loop (k - 1) (xs.push (Float.ofNat i)) (ys.push (Float.ofNat j)) g'
  loop n #[] #[] g

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
  let (xs, _) := sampleUniformNatDet n alphabet ({ state := 0xDEADBEEFCAFEBABE } : Lcg)
  pure xs

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
      let (x, _) := sampleUniformNatDet nSamples alphabet ({ state := 0xABCDEF0123456789 } : Lcg)
      let (y, _) := sampleUniformNatDet nSamples alphabet ({ state := 0x1122334455667788 } : Lcg)
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
      let (x, _) := sampleUniformNatDet nSamples alphabet ({ state := 0x0F0E0D0C0B0A0908 } : Lcg)
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

      let (x, g0) := sampleUniformNatDet nSamples 2 ({ state := 0x1111222233334444 } : Lcg)
      let thresh := (p * 1000000.0).toUInt64.toNat
      let mut g := g0
      let mut y : Array Float := #[]
      for xi in x do
        let (r, g') := g.nextNat 1000000
        g := g'
        let flip := r < thresh
        y := y.push (if flip then 1.0 - xi else xi)

      let truths := truthMap [
        ("I_XY", iXY),
        ("H_Y_given_X", hYgivenX),
        ("H_Y", hY)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Skewed categorical oracle with known entropy. -/
def skewedCategoricalOracle : GenerativeOracle :=
  { name := "skewed_categorical"
  , description := "Skewed categorical distribution with analytic entropy"
  , generate := fun _regime nSamples => do
      let probs := normalizeProbs #[0.5, 0.2, 0.15, 0.1, 0.05]
      let (x, _) := sampleCategoricalDet nSamples probs ({ state := 0x1234567890ABCDEF } : Lcg)
      let h := shannonEntropyCategorical probs
      let truths := truthMap [("H_X", h)]
      pure { bundle := { x := some x }, truths := truths }
  }

/-- Joint categorical oracle with known entropies and mutual information. -/
def jointCategoricalOracle : GenerativeOracle :=
  { name := "joint_categorical"
  , description := "Joint categorical distribution with analytic Hx, Hy, Hxy, MI, NED, NTE"
  , generate := fun _regime nSamples => do
      let joint := #[
        #[0.4, 0.1],
        #[0.1, 0.4]
      ]
      let joint := normalizeJoint joint
      let (x, y, _) := sampleJointCategoricalDet nSamples joint ({ state := 0x0BADF00DCAFED00D } : Lcg)
      let hxy := jointEntropyCategorical joint
      let (px, py) := jointToMarginals joint
      let hx := shannonEntropyCategorical px
      let hy := shannonEntropyCategorical py
      let mi := hx + hy - hxy
      let minH := if hx ≤ hy then hx else hy
      let maxH := if hx ≥ hy then hx else hy
      let ned := if maxH == 0.0 then 0.0 else (hxy - minH) / maxH
      let nte := if maxH == 0.0 then 0.0 else ((2.0 * hxy - hx - hy) / maxH)
      let truths := truthMap [
        ("H_X", hx), ("H_Y", hy), ("H_XY", hxy), ("I_XY", mi),
        ("NED", ned), ("NTE", nte)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Pair of categorical distributions for KL/JS/TVD/XE checks. -/
def pairDistributionsOracle : GenerativeOracle :=
  { name := "pair_distributions"
  , description := "Two categorical distributions with analytic KL/JS/TVD/XE"
  , generate := fun _regime nSamples => do
      let p := normalizeProbs #[0.6, 0.25, 0.1, 0.05]
      let q := normalizeProbs #[0.4, 0.35, 0.2, 0.05]
      let (x, g1) := sampleCategoricalDet nSamples p ({ state := 0xA1A2A3A4A5A6A7A8 } : Lcg)
      let (y, _) := sampleCategoricalDet nSamples q g1
      let hP := shannonEntropyCategorical p
      let kl ← match klCategorical p q with
        | .ok v => pure v
        | .error e => throw <| IO.userError e
      let js ← match jsCategorical p q with
        | .ok v => pure v
        | .error e => throw <| IO.userError e
      let tvd ← match tvdCategorical p q with
        | .ok v => pure v
        | .error e => throw <| IO.userError e
      let xe := hP + kl
      let truths := truthMap [
        ("D_KL", kl), ("D_JS", js), ("TVD", tvd), ("H_XE", xe), ("H_X", hP)
      ]
      pure { bundle := { x := some x, y := some y }, truths := truths }
  }

/-- Binary Markov chain oracle with analytic entropy rate. -/
def binaryMarkovOracle (p00 p11 : Float) : GenerativeOracle :=
  { name := "binary_markov"
  , description := "Binary Markov chain with analytic entropy rate"
  , generate := fun _regime nSamples => do
      let denom := (2.0 - p00 - p11)
      let pi0 := if denom == 0.0 then 0.5 else (1.0 - p11) / denom
      let pi1 := 1.0 - pi0
      let h := pi0 * shannonEntropyBernoulli (1.0 - p00) + pi1 * shannonEntropyBernoulli (1.0 - p11)
      -- Generate samples
      let mut g : Lcg := { state := 0xFEEDFACECAFEBEEF }
      let (u0, g0) := Lcg.nextFloat01 g
      let mut state := if u0 ≤ pi0 then 0 else 1
      g := g0
      let mut xs : Array Float := #[]
      for _ in [:nSamples] do
        xs := xs.push (Float.ofNat state)
        let (u, g') := Lcg.nextFloat01 g
        g := g'
        if state == 0 then
          state := if u ≤ p00 then 0 else 1
        else
          state := if u ≤ p11 then 1 else 0
      let truths := truthMap [("H_RATE", h)]
      pure { bundle := { x := some xs }, truths := truths }
  }

/-- Binary order-2 Markov chain oracle with analytic entropy rate. -/
def binaryMarkov2Oracle (p00 p01 p10 p11 : Float) : GenerativeOracle :=
  { name := "binary_markov_order2"
  , description := "Binary order-2 Markov chain with analytic entropy rate"
  , generate := fun _regime nSamples => do
      let probs : Array Float := #[p00, p01, p10, p11] -- P(next=1 | 00,01,10,11)
      -- Stationary distribution over pair states via power iteration
      let mut pi : Array Float := #[0.25, 0.25, 0.25, 0.25]
      for _ in [:2000] do
        let mut next : Array Float := #[0.0, 0.0, 0.0, 0.0]
        for i in [:4] do
          let p1 := probs.get! i
          let p0 := 1.0 - p1
          let b := i % 2
          let idx0 := b * 2
          let idx1 := b * 2 + 1
          next := next.set! idx0 (next.get! idx0 + (pi.get! i) * p0)
          next := next.set! idx1 (next.get! idx1 + (pi.get! i) * p1)
        pi := next
      let s := pi.foldl (init := 0.0) (· + ·)
      if s > 0.0 then
        pi := pi.map (fun v => v / s)
      let h :=
        (List.range 4).foldl (init := 0.0) (fun acc i =>
          acc + (pi.get! i) * shannonEntropyBernoulli (probs.get! i))

      -- Sample initial pair from stationary distribution
      let cdf : Array Float := Id.run do
        let mut acc := 0.0
        let mut out := #[]
        for p in pi do
          acc := acc + p
          out := out.push acc
        return out
      let rec findIdx (xs : List Float) (u : Float) (i : Nat) : Nat :=
        match xs with
        | [] => 3
        | x :: xs => if u ≤ x then i else findIdx xs u (i + 1)
      let mut g : Lcg := { state := 0xCAFEBABEDEADC0DE }
      let (u0, g0) := Lcg.nextFloat01 g
      g := g0
      let idx := findIdx cdf.toList u0 0
      let mut a : Nat := idx / 2
      let mut b : Nat := idx % 2

      let mut xs : Array Float := #[]
      if nSamples == 0 then
        xs := #[]
      else if nSamples == 1 then
        xs := xs.push (Float.ofNat a)
      else
        xs := xs.push (Float.ofNat a)
        xs := xs.push (Float.ofNat b)
        for _ in [0:(nSamples - 2)] do
          let p1 := probs.get! (a * 2 + b)
          let (u, g') := Lcg.nextFloat01 g
          g := g'
          let next := if u ≤ p1 then 1 else 0
          xs := xs.push (Float.ofNat next)
          a := b
          b := next

      let truths := truthMap [("H_RATE", h)]
      pure { bundle := { x := some xs }, truths := truths }
  }

/-- Markov chain X→Y→Z for data processing checks. -/
def markovChainOracle (flip1 flip2 : Float) : GenerativeOracle :=
  { name := "markov_chain_xyz"
  , description := "Binary Markov chain X->Y->Z via BSC"
  , generate := fun _regime nSamples => do
      let mut g : Lcg := { state := 0x13579BDF2468ACE0 }
      let (x, g1) := sampleUniformNatDet nSamples 2 g
      g := g1
      let mut y : Array Float := #[]
      let mut z : Array Float := #[]
      for xi in x do
        let (u1, g2) := Lcg.nextFloat01 g
        g := g2
        let yi := if u1 ≤ flip1 then 1.0 - xi else xi
        y := y.push yi
        let (u2, g3) := Lcg.nextFloat01 g
        g := g3
        let zi := if u2 ≤ flip2 then 1.0 - yi else yi
        z := z.push zi
      let truths := truthMap []
      pure { bundle := { x := some x, y := some y, z := some z }, truths := truths }
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
        match vals["I_XY"]? with
        | some v => v ≥ 0.0
        | none => true
    , explanation := "Mutual information must be non-negative"
    }
  , { name := "kl_nonnegative"
    , check := fun vals =>
        match vals["D_KL"]? with
        | some v => v ≥ 0.0
        | none => true
    , explanation := "KL divergence must be non-negative"
    }
  , { name := "entropy_bound"
    , check := fun vals =>
        match vals["H_X"]? with
        | some h =>
          match vals["alphabet_size"]? with
          | some a => h ≤ log2 a
          | none => true
        | none => true
    , explanation := "Entropy cannot exceed log(|alphabet|)"
    }
  ]

end ITE
