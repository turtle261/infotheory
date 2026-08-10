import Std
import ITE.Types
import ITE.Estimators

namespace ITE

open Std
open IO
open System

/-!
# Bitwise Shannon measures (Lean reference + Rust CLI validation)

Lean-side authority for pooled binary-alphabet empirical estimators (`*_bits`)
and the algorithmic `*_per_bit` unit conversion (`rate_per_bit = rate_per_byte / 8`).

Bits from every byte are pooled without regard to position within the byte.
This matches `infotheory::api::bit_histogram` / `joint_bit_histogram`.

This module intentionally imports only `Estimators` (not `Oracles`/`Runner`) so
its artifacts stay small enough to build and run under a memory budget.
-/

/-- Count set bits in a single byte (Nat in `0..8`). -/
def u8Popcount (b : UInt8) : Nat :=
  Id.run do
    let mut n : Nat := 0
    let mut x : Nat := b.toNat
    for _ in [:8] do
      n := n + x % 2
      x := x / 2
    return n

/-- Pooled marginal bit histogram `[P(0), P(1)]` over all `8N` bits. -/
def bitHistogram (data : ByteArray) : Array Float :=
  Id.run do
    let mut ones : Nat := 0
    for i in [:data.size] do
      ones := ones + u8Popcount (data.get! i)
    let total := data.size * 8
    if total == 0 then
      return #[0.0, 0.0]
    let p1 := (Float.ofNat ones) / (Float.ofNat total)
    return #[1.0 - p1, p1]

/-- Pooled joint bit histogram `[P(0,0), P(0,1), P(1,0), P(1,1)]` over aligned prefixes. -/
def jointBitHistogram (x y : ByteArray) : Array Float :=
  Id.run do
    let n := Nat.min x.size y.size
    let mut c11 : Nat := 0
    let mut c10 : Nat := 0
    let mut c01 : Nat := 0
    for i in [:n] do
      let bx := (x.get! i).toNat
      let yb := (y.get! i).toNat
      for bit in [:8] do
        let xi := (bx >>> bit) % 2
        let yi := (yb >>> bit) % 2
        if xi == 1 && yi == 1 then
          c11 := c11 + 1
        else if xi == 1 && yi == 0 then
          c10 := c10 + 1
        else if xi == 0 && yi == 1 then
          c01 := c01 + 1
    let total := n * 8
    if total == 0 then
      return #[0.0, 0.0, 0.0, 0.0]
    let c00 := total - c11 - c10 - c01
    let t := Float.ofNat total
    return #[
      (Float.ofNat c00) / t,
      (Float.ofNat c01) / t,
      (Float.ofNat c10) / t,
      (Float.ofNat c11) / t
    ]

/-- Shannon entropy of a finite probability vector (skipping zero bins). -/
def shannonOfProbs (probs : Array Float) : Float :=
  probs.foldl (init := 0.0) (fun acc p =>
    if p ≤ 0.0 then acc else acc - p * (Float.log p / Float.log 2.0))

/-- Order-0 empirical Shannon entropy `H₀,bits` over the pooled binary alphabet. -/
def empiricalEntropyBits (data : ByteArray) : Float :=
  shannonOfProbs (bitHistogram data)

/-- Order-0 empirical joint Shannon entropy `H₀,bits(X,Y)` over aligned bit pairs. -/
def empiricalJointEntropyBits (x y : ByteArray) : Float :=
  shannonOfProbs (jointBitHistogram x y)

/-- Order-0 empirical mutual information `I₀,bits(X;Y)`. -/
def empiricalMutualInformationBits (x y : ByteArray) : Float :=
  let n := Nat.min x.size y.size
  let x' := x.extract 0 n
  let y' := y.extract 0 n
  let h := empiricalEntropyBits x' + empiricalEntropyBits y' - empiricalJointEntropyBits x' y'
  if h < 0.0 then 0.0 else h

/-- Total variation distance between pooled bit distributions. -/
def tvdBits (x y : ByteArray) : Float :=
  if x.size == 0 || y.size == 0 then 0.0
  else
    let p := bitHistogram x
    let q := bitHistogram y
    let s := Float.abs (p.get! 0 - q.get! 0) + Float.abs (p.get! 1 - q.get! 1)
    let v := s / 2.0
    if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v

private def floorEps (q : Float) : Float :=
  if q < 1e-12 then 1e-12 else q

/-- KL divergence `D_KL(P||Q)` over pooled bit distributions, with `1e-12` floor on Q. -/
def dKlBits (x y : ByteArray) : Float :=
  if x.size == 0 || y.size == 0 then 0.0
  else
    let p := bitHistogram x
    let q := bitHistogram y
    let acc := Id.run do
      let mut acc := 0.0
      for i in [:2] do
        let pi := p.get! i
        if pi > 0.0 then
          let qi := floorEps (q.get! i)
          acc := acc + pi * (Float.log (pi / qi) / Float.log 2.0)
      return acc
    if acc < 0.0 then 0.0 else acc

/-- Jensen–Shannon divergence over pooled bit distributions (bits; bounded by 1). -/
def jsDivBits (x y : ByteArray) : Float :=
  if x.size == 0 || y.size == 0 then 0.0
  else
    let p := bitHistogram x
    let q := bitHistogram y
    let m0 := 0.5 * (p.get! 0 + q.get! 0)
    let m1 := 0.5 * (p.get! 1 + q.get! 1)
    let m := #[m0, m1]
    let jsd := Id.run do
      let mut klPm := 0.0
      let mut klQm := 0.0
      for i in [:2] do
        let pi := p.get! i
        let qi := q.get! i
        let mi := m.get! i
        if pi > 0.0 then
          klPm := klPm + pi * (Float.log (pi / mi) / Float.log 2.0)
        if qi > 0.0 then
          klQm := klQm + qi * (Float.log (qi / mi) / Float.log 2.0)
      return 0.5 * klPm + 0.5 * klQm
    if jsd < 0.0 then 0.0 else if jsd > 1.0 then 1.0 else jsd

/-- Cross-entropy of test bit distribution under train bit distribution. -/
def empiricalCrossEntropyBits (testData trainData : ByteArray) : Float :=
  if testData.size == 0 then 0.0
  else
    let p := bitHistogram testData
    let q := bitHistogram trainData
    Id.run do
      let mut h := 0.0
      for i in [:2] do
        let pi := p.get! i
        if pi > 0.0 then
          let qi := floorEps (q.get! i)
          h := h - pi * (Float.log qi / Float.log 2.0)
      return h

/-- Call the Rust CLI with an explicit primitive name. -/
def runInfotheoryPrimitive
    (binPath : FilePath)
    (primName : String)
    (paths : Array FilePath)
    (params : EstimatorParams := defaultParams) : IO (Except String Float) := do
  let paramString (k : String) : Option String := params.strings[k]?
  let mut args : Array String := #[primName]
  for p in paths do
    args := args.push p.toString
  match paramString "rate_backend" with
  | some rb => args := args.push "--rate-backend" |>.push rb
  | none => pure ()
  match paramString "compression_backend" with
  | some nb => args := args.push "--compression-backend" |>.push nb
  | none => pure ()
  match paramString "method" with
  | some m => args := args.push "--method" |>.push m
  | none => pure ()
  let out ← IO.Process.output { cmd := binPath.toString, args := args }
  if out.exitCode ≠ 0 then
    return .error s!"infotheory call failed ({primName}): {out.stderr}"
  return parseFloatSimple out.stdout

/-- Absolute-error check of a Rust `_bits` CLI primitive against a Lean reference. -/
def checkBitsUnary
    (binPath : FilePath)
    (label : String)
    (prim : String)
    (data : ByteArray)
    (leanVal : Float)
    (tol : Float := 1e-9) : IO Bool := do
  let path ← writeTempBytes "bits_x" data
  match ← runInfotheoryPrimitive binPath prim #[path] with
  | .error e =>
    IO.println s!"[BITS] FAIL {label}: {e}"
    pure false
  | .ok rustVal =>
    let err := Float.abs (rustVal - leanVal)
    if err > tol then
      IO.println s!"[BITS] FAIL {label}: rust={rustVal} lean={leanVal} absErr={err} tol={tol}"
      pure false
    else
      IO.println s!"[BITS] PASS {label}: rust={rustVal} lean={leanVal}"
      pure true

/-- Absolute-error check for a binary Rust `_bits` CLI primitive. -/
def checkBitsBinary
    (binPath : FilePath)
    (label : String)
    (prim : String)
    (x y : ByteArray)
    (leanVal : Float)
    (tol : Float := 1e-9) : IO Bool := do
  let px ← writeTempBytes "bits_x" x
  let py ← writeTempBytes "bits_y" y
  match ← runInfotheoryPrimitive binPath prim #[px, py] with
  | .error e =>
    IO.println s!"[BITS] FAIL {label}: {e}"
    pure false
  | .ok rustVal =>
    let err := Float.abs (rustVal - leanVal)
    if err > tol then
      IO.println s!"[BITS] FAIL {label}: rust={rustVal} lean={leanVal} absErr={err} tol={tol}"
      pure false
    else
      IO.println s!"[BITS] PASS {label}: rust={rustVal} lean={leanVal}"
      pure true

/-- Fill `n` bytes with a constant value. -/
def replicateByte (n : Nat) (b : UInt8) : ByteArray :=
  Id.run do
    let mut out := ByteArray.empty
    for _ in [:n] do
      out := out.push b
    return out

/-- Fill `n` bytes using `IO.rand` (avoids importing Oracles/Lcg). -/
def randomBytesIO (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  for _ in [:n] do
    let v ← IO.rand 0 255
    out := out.push (UInt8.ofNat v)
  return out

/-- Ensure the Rust CLI binary exists before validation. -/
def ensureBitsExecutable (path : FilePath) : IO Unit := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"Missing required executable: {path}. Build with `cargo build --release --features cli` and re-run."

/-- Verify Rust `_bits` / `_per_bit` CLI against Lean definitions. -/
def runBitwiseSuite (binPath : FilePath) : IO Bool := do
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "[BITS] Bitwise Shannon Measures Validation"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  let zeros := replicateByte 64 (UInt8.ofNat 0)
  let ones := replicateByte 64 (UInt8.ofNat 255)
  let mixed ← randomBytesIO 128
  let ascii : ByteArray := "The quick brown fox jumps over the lazy dog".toUTF8

  let mut ok := true

  if !(← checkBitsUnary binPath "H0_bits(all-0)" "h_bits" zeros (empiricalEntropyBits zeros)) then
    ok := false
  if !(← checkBitsUnary binPath "H0_bits(all-1)" "h_bits" ones (empiricalEntropyBits ones)) then
    ok := false
  if !(← checkBitsUnary binPath "H0_bits(mixed)" "h_bits" mixed (empiricalEntropyBits mixed)) then
    ok := false

  let hBitsAscii := empiricalEntropyBits ascii
  match ← runInfotheoryPrimitive binPath "h" #[← writeTempBytes "ascii" ascii] with
  | .error e =>
    ok := false
    IO.println s!"[BITS] FAIL H_bytes(ascii) call: {e}"
  | .ok hBytes =>
    let scaled := hBytes / 8.0
    if Float.abs (hBitsAscii - scaled) ≤ 1e-4 then
      ok := false
      IO.println s!"[BITS] FAIL empirical H_bits must differ from H_bytes/8 on ASCII: bits={hBitsAscii} bytes/8={scaled}"
    else
      IO.println s!"[BITS] PASS empirical H_bits ≠ H_bytes/8 on ASCII (bits={hBitsAscii}, bytes/8={scaled})"

  if !(← checkBitsBinary binPath "TVD_bits(0,1)" "tvd_bits" zeros ones (tvdBits zeros ones)) then
    ok := false
  if !(← checkBitsBinary binPath "JSD_bits(0,1)" "js_bits" zeros ones (jsDivBits zeros ones)) then
    ok := false
  if !(← checkBitsBinary binPath "KL_bits(0,1)" "kl_bits" zeros ones (dKlBits zeros ones) 1e-6) then
    ok := false
  if !(← checkBitsBinary binPath "JSD_bits(x,x)" "js_bits" mixed mixed (jsDivBits mixed mixed)) then
    ok := false

  let y ← randomBytesIO 128
  if !(← checkBitsBinary binPath "H_joint_bits" "joint_entropy_bits" mixed y
      (empiricalJointEntropyBits mixed y)) then
    ok := false
  if !(← checkBitsBinary binPath "I_bits" "mi_bits" mixed y
      (empiricalMutualInformationBits mixed y)) then
    ok := false
  if !(← checkBitsBinary binPath "XE_bits" "xe_bits" mixed y
      (empiricalCrossEntropyBits mixed y)) then
    ok := false

  match ← runInfotheoryPrimitive binPath "mi_bits"
      #[← writeTempBytes "mix" mixed, ← writeTempBytes "y" y] with
  | .error e =>
    ok := false
    IO.println s!"[BITS] FAIL MI invariant call: {e}"
  | .ok mi =>
    let hx := empiricalEntropyBits mixed
    let hy := empiricalEntropyBits y
    let hxy := empiricalJointEntropyBits mixed y
    let minH := if hx < hy then hx else hy
    if mi < -1e-12 then
      ok := false
      IO.println s!"[BITS] FAIL I_bits >= 0: {mi}"
    if hxy > 2.0 + 1e-12 then
      ok := false
      IO.println s!"[BITS] FAIL H_joint_bits <= 2: {hxy}"
    if mi > minH + 1e-9 then
      ok := false
      IO.println s!"[BITS] FAIL I_bits <= min(Hx,Hy): mi={mi} min={minH}"
    else
      IO.println s!"[BITS] PASS MI/joint invariants (mi={mi}, Hxy={hxy})"

  let paramsCtw : EstimatorParams :=
    { scalars := HashMap.empty
    , strings := (HashMap.empty : HashMap String String)
        |>.insert "rate_backend" "ctw"
        |>.insert "method" "16" }
  let ratePath ← writeTempBytes "rate" mixed
  match ← runInfotheoryPrimitive binPath "entropy_rate" #[ratePath] paramsCtw with
  | .error e =>
    ok := false
    IO.println s!"[BITS] FAIL entropy_rate call: {e}"
  | .ok rateBytes =>
    match ← runInfotheoryPrimitive binPath "h_rate_per_bit" #[ratePath] paramsCtw with
    | .error e =>
      ok := false
      IO.println s!"[BITS] FAIL h_rate_per_bit call: {e}"
    | .ok ratePerBit =>
      let expected := rateBytes / 8.0
      let err := Float.abs (ratePerBit - expected)
      if err > 1e-12 then
        ok := false
        IO.println s!"[BITS] FAIL per_bit = per_byte/8: per_bit={ratePerBit} expected={expected}"
      else
        IO.println s!"[BITS] PASS algorithmic per_bit = per_byte/8 ({ratePerBit})"

  if ok then
    IO.println "[BITS] PASS: bitwise Shannon suite"
  pure ok

end ITE
