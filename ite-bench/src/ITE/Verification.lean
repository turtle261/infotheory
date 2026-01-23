import Std
import ITE.Types

namespace ITE

open Std

/-- Helper to run an estimator in `IO`, surfacing errors as exceptions. -/
def runEstimateIO (est : Estimator) (q : Quantity) (bundle : SampleBundle)
  (params : EstimatorParams := defaultParams) : IO Float :=
  do
    match ← est.estimate q bundle params with
    | .ok v => pure v
    | .error e => throw <| IO.userError s!"Estimator {est.name} failed: {e}"

/-- Single metric trial result. -/
structure MetricEntry where
  passed : Bool
  value : Float
  regime : DataRegime
  details : String
  deriving Repr

structure MetricReport where
  nonNegativity : Array MetricEntry
  identity : Array MetricEntry
  symmetry : Array MetricEntry
  triangle : Array MetricEntry
  deriving Repr

/-- Verify metric axioms for an NCD-like quantity using a triplet generator. -/
def verifyMetric (estimator : Estimator)
    (tripletGen : DataRegime → IO (SampleBundle × SampleBundle × SampleBundle))
    (regime : DataRegime)
    (nTrials : Nat := 100)
    (tols : MetricTolerances := ToleranceDefaults.defaults.metric) : IO MetricReport := do
  let mut nn : Array MetricEntry := #[]
  let mut id : Array MetricEntry := #[]
  let mut sym : Array MetricEntry := #[]
  let mut tri : Array MetricEntry := #[]
  for _ in [:nTrials] do
    let (x, y, z) ← tripletGen regime
    let dxy ← runEstimateIO estimator .ncdVitanyi { x := x.x, y := y.x, bytesX := x.bytesX, bytesY := y.bytesX }
    let dxx ← runEstimateIO estimator .ncdVitanyi { x := x.x, y := x.x, bytesX := x.bytesX, bytesY := x.bytesX }
    let dyx ← runEstimateIO estimator .ncdVitanyi { x := y.x, y := x.x, bytesX := y.bytesX, bytesY := x.bytesX }
    let dxz ← runEstimateIO estimator .ncdVitanyi { x := x.x, y := z.x, bytesX := x.bytesX, bytesY := z.bytesX }
    let dyz ← runEstimateIO estimator .ncdVitanyi { x := y.x, y := z.x, bytesX := y.bytesX, bytesY := z.bytesX }
    nn := nn.push { passed := dxy ≥ -tols.nonNegativity, value := dxy, regime := regime, details := "non-negativity" }
    id := id.push { passed := Float.abs dxx ≤ tols.identity, value := dxx, regime := regime, details := "identity" }
    sym := sym.push { passed := Float.abs (dxy - dyx) ≤ tols.symmetry, value := dxy, regime := regime, details := s!"sym d_xy={dxy} d_yx={dyx}" }
    let triHolds := dxz ≤ dxy + dyz + tols.triangle
    tri := tri.push { passed := triHolds, value := dxz, regime := regime, details := s!"triangle d_xz={dxz} d_xy+ d_yz={dxy + dyz}" }
  return { nonNegativity := nn, identity := id, symmetry := sym, triangle := tri }

/-- Violation details for information-theoretic inequalities. -/
structure InequalityViolation where
  id : InequalityId
  regime : DataRegime
  magnitude : Float
  details : String
  deriving Repr

/-- Verify subadditivity, MI non-negativity, and data processing using estimator outputs. -/
def verifyInequalities (estimator : Estimator)
    (dataGen : DataRegime → IO SampleBundle)
    (regime : DataRegime)
    (nTrials : Nat := 100)
    (tol : Float := ToleranceDefaults.defaults.metric.nonNegativity) : IO (Array InequalityViolation) := do
  let mut violations : Array InequalityViolation := #[]
  for _ in [:nTrials] do
    let data ← dataGen regime
    let hX ← runEstimateIO estimator .shannonEntropy { x := data.x }
    let hY ← runEstimateIO estimator .shannonEntropy { x := data.y }
    let hXY ← runEstimateIO estimator .jointEntropy { x := data.x, y := data.y, xy := data.xy }
    let iXY := hX + hY - hXY
    if hXY > hX + hY + tol then
      violations := violations.push { id := .subadditivity, regime := regime, magnitude := hXY - (hX + hY), details := s!"H(XY)={hXY} vs H(X)+H(Y)={hX + hY}" }
    if iXY < -tol then
      violations := violations.push { id := .miNonnegative, regime := regime, magnitude := -iXY, details := s!"I(X;Y)={iXY}" }
    match data.z with
    | some _ =>
      let hZ ← runEstimateIO estimator .shannonEntropy { x := data.z }
      let hXZ ← runEstimateIO estimator .jointEntropy { x := data.x, y := data.z }
      let iXZ := hX + hZ - hXZ
      if iXY + tol < iXZ then
        violations := violations.push { id := .dataProcessing, regime := regime, magnitude := iXZ - iXY, details := s!"I(X;Y)={iXY} < I(X;Z)={iXZ}" }
    | none => pure ()
  return violations

def verifyInequalitiesWith (estimator : Estimator)
    (dataGen : DataRegime → IO SampleBundle)
    (regime : DataRegime)
    (params : EstimatorParams)
    (nTrials : Nat := 100)
    (tol : Float := ToleranceDefaults.defaults.metric.nonNegativity) : IO (Array InequalityViolation) := do
  let mut violations : Array InequalityViolation := #[]
  for _ in [:nTrials] do
    let data ← dataGen regime
    let hX ← runEstimateIO estimator .shannonEntropy { x := data.x } params
    let hY ← runEstimateIO estimator .shannonEntropy { x := data.y } params
    let hXY ← runEstimateIO estimator .jointEntropy { x := data.x, y := data.y, xy := data.xy } params
    let iXY := hX + hY - hXY
    if hXY > hX + hY + tol then
      violations := violations.push { id := .subadditivity, regime := regime, magnitude := hXY - (hX + hY), details := s!"H(XY)={hXY} vs H(X)+H(Y)={hX + hY}" }
    if iXY < -tol then
      violations := violations.push { id := .miNonnegative, regime := regime, magnitude := -iXY, details := s!"I(X;Y)={iXY}" }
    match data.z with
    | some _ =>
      let hZ ← runEstimateIO estimator .shannonEntropy { x := data.z } params
      let hXZ ← runEstimateIO estimator .jointEntropy { x := data.x, y := data.z } params
      let iXZ := hX + hZ - hXZ
      if iXY + tol < iXZ then
        violations := violations.push { id := .dataProcessing, regime := regime, magnitude := iXZ - iXY, details := s!"I(X;Y)={iXY} < I(X;Z)={iXZ}" }
    | none => pure ()
  return violations

/-- Accuracy statistics across trials. -/
structure AccuracyReport where
  mae : Float
  mape : Float
  rmse : Float
  maxAbsError : Float
  maxRelError : Float
  regime : DataRegime
  quantity : Quantity
  deriving Repr

/-- Verify accuracy against oracle values. The `oracleGen` must return data and the oracle truth. -/
def verifyAccuracy (estimator : Estimator)
    (oracleGen : DataRegime → IO (SampleBundle × Float))
    (quantity : Quantity)
    (regime : DataRegime)
    (nTrials : Nat := 100) : IO AccuracyReport := do
  let mut absErrs : Array Float := #[]
  let mut relErrs : Array Float := #[]
  let mut sqErrs : Array Float := #[]
  for _ in [:nTrials] do
    let (data, oracle) ← oracleGen regime
    let est ← runEstimateIO estimator quantity data
    let absErr := Float.abs (est - oracle)
    let relErr := if oracle == 0.0 then 1.0e9 else absErr / Float.abs oracle
    absErrs := absErrs.push absErr
    relErrs := relErrs.push relErr
    sqErrs := sqErrs.push ((est - oracle) * (est - oracle))
  let mean (xs : Array Float) : Float :=
    if xs.isEmpty then 0.0 else xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size
  let rms (xs : Array Float) : Float := Float.sqrt (mean xs)
  let maxVal (xs : Array Float) : Float := xs.foldl (init := 0.0) (fun a b => if a < b then b else a)
  return {
    mae := mean absErrs
    mape := mean relErrs
    rmse := rms sqErrs
    maxAbsError := maxVal absErrs
    maxRelError := maxVal relErrs
    regime := regime
    quantity := quantity
  }

def verifyAccuracyWith (estimator : Estimator)
    (oracleGen : DataRegime → IO (SampleBundle × Float))
    (quantity : Quantity)
    (regime : DataRegime)
    (params : EstimatorParams)
    (nTrials : Nat := 100) : IO AccuracyReport := do
  let mut absErrs : Array Float := #[]
  let mut relErrs : Array Float := #[]
  let mut sqErrs : Array Float := #[]
  for _ in [:nTrials] do
    let (data, oracle) ← oracleGen regime
    let est ← runEstimateIO estimator quantity data params
    let absErr := Float.abs (est - oracle)
    let relErr := if oracle == 0.0 then 1.0e9 else absErr / Float.abs oracle
    absErrs := absErrs.push absErr
    relErrs := relErrs.push relErr
    sqErrs := sqErrs.push ((est - oracle) * (est - oracle))
  let mean (xs : Array Float) : Float :=
    if xs.isEmpty then 0.0 else xs.foldl (init := 0.0) (· + ·) / Float.ofNat xs.size
  let rms (xs : Array Float) : Float := Float.sqrt (mean xs)
  let maxVal (xs : Array Float) : Float := xs.foldl (init := 0.0) (fun a b => if a < b then b else a)
  return {
    mae := mean absErrs
    mape := mean relErrs
    rmse := rms sqErrs
    maxAbsError := maxVal absErrs
    maxRelError := maxVal relErrs
    regime := regime
    quantity := quantity
  }

end ITE
