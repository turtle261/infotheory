import ITE.Types
import ITE.Oracles
import ITE.Verification
import ITE.Estimators
import ITE.Sequitur

open ITE
open Std
open IO
open System

set_option maxRecDepth 2000

namespace ITE

def ensureExecutable (path : FilePath) : IO Unit := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"Missing required executable: {path}. Build the Rust workspace (e.g. `cargo build --release`) and re-run."

private def randBytes (n : Nat) : IO ByteArray := do
  let mut out := ByteArray.empty
  for _ in [:n] do
    let v ← IO.rand 0 255
    out := out.push (UInt8.ofNat v)
  return out

private def mutateBytes (bytes : ByteArray) (nFlips : Nat) : IO ByteArray := do
  if bytes.size == 0 then
    return bytes
  let mut out := bytes
  for _ in [:nFlips] do
    let idx ← IO.rand 0 (bytes.size - 1)
    let v ← IO.rand 0 255
    out := out.set! idx (UInt8.ofNat v)
  return out

/-- Triplet generator for metric checks on NCD-like quantities.

We generate a base sequence `x`, then two slightly mutated variants `y` and `z`.
This tends to make triangle/symmetry violations easier to detect if there are
bugs (e.g. asymmetric concatenation, non-deterministic compression settings).
-/
private def ncdTripletGen (regime : DataRegime) : IO (SampleBundle × SampleBundle × SampleBundle) := do
  let len := match regime.sampleSize with
    | .tiny => 128
    | .small => 512
    | .medium => 2048
    | .large => 8192
    | .asymptotic => 16384

  let base ← randBytes len
  let flips := match regime.compressibility with
    | .high | .structured => 1
    | .medium => 4
    | .low => 16
    | .incompressible => 64

  let y ← mutateBytes base flips
  let z ← mutateBytes base flips

  let bx : SampleBundle := { bytesX := some base }
  let byBundle : SampleBundle := { bytesX := some y }
  let bz : SampleBundle := { bytesX := some z }
  return (bx, byBundle, bz)

private def validateSequiturDomain
    (binPath : FilePath)
    (label : String)
    (inputs : Array ByteArray)
    (alphabetPrefix : Nat)
    (chunkSize : Nat := 128) : IO Bool := do
  IO.println s!"[SEQUITUR] Validating {label} domain ({inputs.size} inputs, chunk={chunkSize})"
  let chunks := batchHexInputs inputs chunkSize
  let mut ok := true
  for chunk in chunks do
    match ← runRustSequiturDebug binPath chunk 64 alphabetPrefix with
    | .error e =>
      ok := false
      IO.println s!"[SEQUITUR] FAIL: {e}"
    | .ok batch =>
      if batch.cases.size != chunk.size then
        ok := false
        IO.println s!"[SEQUITUR] FAIL: expected {chunk.size} cases, got {batch.cases.size}"
      else
        match validateDebugBatch batch with
        | .error e =>
          ok := false
          IO.println s!"[SEQUITUR] FAIL: {e}"
        | .ok _ => pure ()
  if ok then
    IO.println s!"[SEQUITUR] PASS: {label}"
  pure ok

def runSequiturSuite (binPath : FilePath) : IO Bool := do
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "[SEQUITUR] Canonical Grammar Validation"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let binaryInputs := allWordsUpTo #[0, 1] 12
  let ternaryInputs := allWordsUpTo #[0, 1, 2] 9
  let okBinary ← validateSequiturDomain binPath "binary<=12" binaryInputs 4
  let okTernary ← validateSequiturDomain binPath "ternary<=9" ternaryInputs 4
  pure <| okBinary && okTernary

private def oracleGenFromOutcome (key : String) (outcome : OracleOutcome) : IO (SampleBundle × Float) := do
  let some v := outcome.truths[key]?
    | throw <| IO.userError s!"Missing oracle truth key: {key}"
  return (outcome.bundle, v)

private def mkParams
    (maxOrder : Option String := none)
    (rateBackend : Option String := none)
    (ncdBackend : Option String := none)
    (method : Option String := none) : EstimatorParams :=
  Id.run do
    let mut strings := HashMap.empty
    match maxOrder with
    | some v => strings := strings.insert "max_order" v
    | none => pure ()
    match rateBackend with
    | some v => strings := strings.insert "rate_backend" v
    | none => pure ()
    match ncdBackend with
    | some v => strings := strings.insert "ncd_backend" v
    | none => pure ()
    match method with
    | some v => strings := strings.insert "method" v
    | none => pure ()
    return { scalars := HashMap.empty, strings := strings }

private def runSuite : IO Bool := do
  let est := infotheoryEstimator
  let rustBin := FilePath.mk "../target/release/infotheory"
  ensureExecutable rustBin

  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║    ITE Benchmark: Self-Contained Mathematical Validation     ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"
  IO.println ""
  IO.println s!"[INIT] Estimator: {est.name}"
  IO.println ""

  let regimes : Array DataRegime := #[(
    { sampleSize := .small
      alphabetSize := .small
      distribution := .uniform
      dependence := .independent
      dimensionality := .bivariate
      compressibility := .medium
      stationarity := .iid
    }), (
    { sampleSize := .medium
      alphabetSize := .binary
      distribution := .uniform
      dependence := .independent
      dimensionality := .bivariate
      compressibility := .structured
      stationarity := .iid
    }), (
    { sampleSize := .large
      alphabetSize := .small
      distribution := .highlySkewed
      dependence := .independent
      dimensionality := .univariate
      compressibility := .low
      stationarity := .iid
    }), (
    { sampleSize := .large
      alphabetSize := .binary
      distribution := .structured
      dependence := .moderate
      dimensionality := .bivariate
      compressibility := .structured
      stationarity := .markov
    })]

  let mut ok := true
  let tolMetric := ToleranceDefaults.defaults.metric
  let strictScale : Float := 0.5
  -- NCD is only approximately a metric at finite sizes (compressor headers, non-idealities).
  -- 10% tolerance for identity checks, 5% for symmetry/triangle.
  let tolMetricNcd : MetricTolerances :=
    { tolMetric with
        identity := 0.10
        symmetry := 0.05
        triangle := 0.05 }

  for (idx, r) in regimes.toList.enum do
    IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    IO.println s!"[REGIME {idx+1}] {repr r}"
    IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    -- Accuracy: independent uniform sources
    let outcomeInd ← independentSourcesOracle.generate r 2000
    let (bundleH, truthHX) ← oracleGenFromOutcome "H_X" outcomeInd
    let (bundleMI, truthMI) ← oracleGenFromOutcome "I_XY" outcomeInd

    let paramsMarg := mkParams (some "0")
    let repHX ← verifyAccuracyWith est (fun _ => pure (bundleH, truthHX)) .shannonEntropy r paramsMarg 30
    let repMI ← verifyAccuracyWith est (fun _ => pure (bundleMI, truthMI)) .mutualInformation r paramsMarg 30

    let tolHX := ToleranceDefaults.defaults.quantity .shannonEntropy r
    let tolMI := ToleranceDefaults.defaults.quantity .mutualInformation r

    IO.println s!"[ACCURACY] H(X) MAE={repHX.mae} maxAbs={repHX.maxAbsError} (tol≈{tolHX})"
    IO.println s!"[ACCURACY] I(X;Y) MAE={repMI.mae} maxAbs={repMI.maxAbsError} (tol≈{tolMI})"

    if repHX.maxAbsError > strictScale * tolHX then
      ok := false
      IO.println "[FAIL] H(X) exceeded tolerance"
    if repMI.maxAbsError > strictScale * tolMI then
      ok := false
      IO.println "[FAIL] I(X;Y) exceeded tolerance"

    -- Inequalities: subadditivity + MI non-negativity (and data processing if Z present)
    let viols ← verifyInequalitiesWith est (fun _ => pure outcomeInd.bundle) r paramsMarg 30 tolMetric.nonNegativity
    if viols.isEmpty then
      IO.println "[INEQ] PASS (no violations in trials)"
    else
      ok := false
      IO.println s!"[INEQ] FAIL ({viols.size} violations)"
      for v in viols do
        IO.println s!"  - {repr v.id}: {v.details} (magnitude={v.magnitude})"

    -- Conditional entropy: if Y = f(X) with f injective, then H(X|Y)=0.
    -- Use identity so this is true for any discrete alphabet.
    let detFn := fun (x : Float) => x
    let outcomeDet ← (deterministicFunctionOracle detFn).generate r 1500
    let (bundleCE, truthCE) ← oracleGenFromOutcome "H_X_given_Y" outcomeDet
    let repCE ← verifyAccuracyWith est (fun _ => pure (bundleCE, truthCE)) .conditionalEntropy r paramsMarg 30
    let tolCE := ToleranceDefaults.defaults.quantity .conditionalEntropy r
    IO.println s!"[ACCURACY] H(X|Y) MAE={repCE.mae} maxAbs={repCE.maxAbsError} (tol≈{tolCE})"
    if repCE.maxAbsError > strictScale * tolCE then
      ok := false
      IO.println "[FAIL] H(X|Y) exceeded tolerance"

    -- Noisy channel: analytic MI
    let outcomeCh ← (noisyChannelOracle 0.1).generate r 2500
    let (bundleCh, truthCh) ← oracleGenFromOutcome "I_XY" outcomeCh
    let repCh ← verifyAccuracyWith est (fun _ => pure (bundleCh, truthCh)) .mutualInformation r paramsMarg 30
    let tolCh := ToleranceDefaults.defaults.quantity .mutualInformation r
    IO.println s!"[ACCURACY] BSC(0.1) I(X;Y) MAE={repCh.mae} maxAbs={repCh.maxAbsError} (tol≈{tolCh})"
    if repCh.maxAbsError > strictScale * tolCh then
      ok := false
      IO.println "[FAIL] Channel MI exceeded tolerance"

    -- Skewed categorical entropy
    let outcomeSkew ← skewedCategoricalOracle.generate r 30000
    let (bundleSkew, truthSkew) ← oracleGenFromOutcome "H_X" outcomeSkew
    let repSkew ← verifyAccuracyWith est (fun _ => pure (bundleSkew, truthSkew)) .shannonEntropy r paramsMarg 20
    let tolSkew := ToleranceDefaults.defaults.quantity .shannonEntropy r
    IO.println s!"[ACCURACY] Skewed H(X) MAE={repSkew.mae} maxAbs={repSkew.maxAbsError} (tol≈{tolSkew})"
    if repSkew.maxAbsError > strictScale * tolSkew then
      ok := false
      IO.println "[FAIL] Skewed H(X) exceeded tolerance"

    -- Joint categorical: Hx, Hy, Hxy, MI, NED, NTE
    let outcomeJoint ← jointCategoricalOracle.generate r 30000
    let (bundleJX, truthJX) ← oracleGenFromOutcome "H_X" outcomeJoint
    let (bundleJY, truthJY) ← oracleGenFromOutcome "H_Y" outcomeJoint
    let (bundleJXY, truthJXY) ← oracleGenFromOutcome "H_XY" outcomeJoint
    let (bundleJMI, truthJMI) ← oracleGenFromOutcome "I_XY" outcomeJoint
    let (bundleJNED, truthJNED) ← oracleGenFromOutcome "NED" outcomeJoint
    let (bundleJNTE, truthJNTE) ← oracleGenFromOutcome "NTE" outcomeJoint

    let repJX ← verifyAccuracyWith est (fun _ => pure (bundleJX, truthJX)) .shannonEntropy r paramsMarg 20
    let repJY ← verifyAccuracyWith est (fun _ => pure (bundleJY, truthJY)) .shannonEntropy r paramsMarg 20
    let repJXY ← verifyAccuracyWith est (fun _ => pure (bundleJXY, truthJXY)) .jointEntropy r paramsMarg 20
    let repJMI ← verifyAccuracyWith est (fun _ => pure (bundleJMI, truthJMI)) .mutualInformation r paramsMarg 20
    let repJNED ← verifyAccuracyWith est (fun _ => pure (bundleJNED, truthJNED)) .ned r paramsMarg 20
    let repJNTE ← verifyAccuracyWith est (fun _ => pure (bundleJNTE, truthJNTE)) .nte r paramsMarg 20

    let tolJ := ToleranceDefaults.defaults.quantity .shannonEntropy r
    if repJX.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] Joint H(X) exceeded tolerance"
    if repJY.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] Joint H(Y) exceeded tolerance"
    if repJXY.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] Joint H(X,Y) exceeded tolerance"
    if repJMI.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] Joint MI exceeded tolerance"
    if repJNED.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] NED exceeded tolerance"
    if repJNTE.maxAbsError > strictScale * tolJ then
      ok := false
      IO.println "[FAIL] NTE exceeded tolerance"

    -- KL / JS / TVD / Cross-Entropy
    let outcomePair ← pairDistributionsOracle.generate r 40000
    let (bundleKL, truthKL) ← oracleGenFromOutcome "D_KL" outcomePair
    let (bundleJS, truthJS) ← oracleGenFromOutcome "D_JS" outcomePair
    let (bundleTVD, truthTVD) ← oracleGenFromOutcome "TVD" outcomePair
    let (bundleXE, truthXE) ← oracleGenFromOutcome "H_XE" outcomePair
    let (_bundleHP, _truthHP) ← oracleGenFromOutcome "H_X" outcomePair

    let repKL ← verifyAccuracyWith est (fun _ => pure (bundleKL, truthKL)) .klDivergence r paramsMarg 20
    let repJS ← verifyAccuracyWith est (fun _ => pure (bundleJS, truthJS)) .jsDivergence r paramsMarg 20
    let repTVD ← verifyAccuracyWith est (fun _ => pure (bundleTVD, truthTVD)) .tvd r paramsMarg 20
    let repXE ← verifyAccuracyWith est (fun _ => pure (bundleXE, truthXE)) .crossEntropy r paramsMarg 20

    let tolKL := ToleranceDefaults.defaults.quantity .klDivergence r
    if repKL.maxAbsError > strictScale * tolKL then
      ok := false
      IO.println "[FAIL] KL exceeded tolerance"
    if repJS.maxAbsError > strictScale * tolKL then
      ok := false
      IO.println "[FAIL] JS exceeded tolerance"
    if repTVD.maxAbsError > strictScale * tolKL then
      ok := false
      IO.println "[FAIL] TVD exceeded tolerance"
    if repXE.maxAbsError > strictScale * tolKL then
      ok := false
      IO.println "[FAIL] Cross-Entropy exceeded tolerance"

    -- Sanity: cross-entropy >= entropy
    let estXE ← runEstimateIO est .crossEntropy outcomePair.bundle paramsMarg
    let estHP ← runEstimateIO est .shannonEntropy outcomePair.bundle paramsMarg
    if estXE + 1e-9 < estHP then
      ok := false
      IO.println s!"[FAIL] Cross-entropy < entropy: {estXE} < {estHP}"

    -- Entropy rate: binary Markov chain
    let outcomeMarkov ← (binaryMarkovOracle 0.9 0.8).generate r 60000
    let (bundleRate, truthRate) ← oracleGenFromOutcome "H_RATE" outcomeMarkov
    let paramsRate := mkParams (some "-1")
    let repRate ← verifyAccuracyWith est (fun _ => pure (bundleRate, truthRate)) .entropyRate r paramsRate 20
    let tolRate := ToleranceDefaults.defaults.quantity .entropyRate r
    IO.println s!"[ACCURACY] Markov H_rate MAE={repRate.mae} maxAbs={repRate.maxAbsError} (tol={tolRate}, strictScale={strictScale}, allowed={strictScale*tolRate})"
    if repRate.maxAbsError > strictScale * tolRate then
      ok := false
      IO.println "[FAIL] Entropy rate exceeded tolerance"

    -- Entropy rate with CTW backend and explicit depth
    let paramsRateCtw := mkParams (some "-1") (some "ctw") none (some "16")
    let repRateCtw ← verifyAccuracyWith est (fun _ => pure (bundleRate, truthRate)) .entropyRate r paramsRateCtw 10
    IO.println s!"[ACCURACY] Markov H_rate (CTW) MAE={repRateCtw.mae} maxAbs={repRateCtw.maxAbsError} (tol={tolRate}, strictScale={strictScale}, allowed={strictScale*tolRate})"
    if repRateCtw.maxAbsError > strictScale * tolRate then
      ok := false
      IO.println "[FAIL] Entropy rate (CTW) exceeded tolerance"

    -- Entropy rate: binary Markov chain (order 2)
    let outcomeMarkov2 ← (binaryMarkov2Oracle 0.1 0.7 0.4 0.9).generate r 60000
    let (bundleRate2, truthRate2) ← oracleGenFromOutcome "H_RATE" outcomeMarkov2
    let repRate2 ← verifyAccuracyWith est (fun _ => pure (bundleRate2, truthRate2)) .entropyRate r paramsRate 20
    IO.println s!"[ACCURACY] Markov2 H_rate MAE={repRate2.mae} maxAbs={repRate2.maxAbsError} (tol={tolRate}, strictScale={strictScale}, allowed={strictScale*tolRate})"
    if repRate2.maxAbsError > strictScale * tolRate then
      ok := false
      IO.println "[FAIL] Entropy rate (Markov2) exceeded tolerance"

    let repRate2Ctw ← verifyAccuracyWith est (fun _ => pure (bundleRate2, truthRate2)) .entropyRate r paramsRateCtw 10
    IO.println s!"[ACCURACY] Markov2 H_rate (CTW) MAE={repRate2Ctw.mae} maxAbs={repRate2Ctw.maxAbsError} (tol={tolRate}, strictScale={strictScale}, allowed={strictScale*tolRate})"
    if repRate2Ctw.maxAbsError > strictScale * tolRate then
      ok := false
      IO.println "[FAIL] Entropy rate (Markov2, CTW) exceeded tolerance"

    -- ZPAQ rate backend sanity: copy-like data should compress well
    let pattern ← randBytes 64
    let reps := 1024
    let mut copyData := ByteArray.empty
    for _ in [:reps] do
      for i in [:pattern.size] do
        copyData := copyData.push (pattern.get! i)
    let copyBundle : SampleBundle := { bytesX := some copyData }
    let paramsZpaq := mkParams (some "-1") (some "zpaq") none (some "2")
    let zpaqRate ← runEstimateIO est .entropyRate copyBundle paramsZpaq
    IO.println s!"[ACCURACY] ZPAQ H_rate on copy-like data = {zpaqRate}"
    if zpaqRate > 0.3 then
      ok := false
      IO.println "[FAIL] ZPAQ entropy rate too high on copy-like data"

    -- Data processing checks using X->Y->Z
    let outcomeXYZ ← (markovChainOracle 0.1 0.2).generate r 30000
    let violsXYZ ← verifyInequalitiesWith est (fun _ => pure outcomeXYZ.bundle) r paramsMarg 20 tolMetric.nonNegativity
    if violsXYZ.isEmpty then
      IO.println "[INEQ] PASS (data processing with Z)"
    else
      ok := false
      IO.println s!"[INEQ] FAIL (data processing) {violsXYZ.size} violations"

    -- Metric checks: NCD (Vitányi) should satisfy metric axioms within tolerance
    IO.println "[METRIC] Checking NCD metric axioms..."
    let metricRep ← verifyMetric est ncdTripletGen r 40 tolMetricNcd
    let passRate (xs : Array MetricEntry) : Float :=
      if xs.isEmpty then 1.0 else
        Float.ofNat (xs.filter (·.passed)).size / Float.ofNat xs.size

    let nnRate := passRate metricRep.nonNegativity
    let idRate := passRate metricRep.identity
    let symRate := passRate metricRep.symmetry
    let triRate := passRate metricRep.triangle

    IO.println s!"  non-negativity pass rate: {nnRate}"
    IO.println s!"  identity pass rate:       {idRate}"
    IO.println s!"  symmetry pass rate:       {symRate}"
    IO.println s!"  triangle pass rate:       {triRate}"

    if nnRate < 0.995 || idRate < 0.92 || symRate < 0.92 || triRate < 0.85 then
      ok := false
      IO.println "[FAIL] Metric axiom pass-rates too low"

    IO.println ""

  let okSequitur ← runSequiturSuite rustBin
  if !okSequitur then
    ok := false

  return ok

end ITE


def main (args : List String) : IO UInt32 := do
  let ok ←
    if args == ["sequitur"] then
      let rustBin := FilePath.mk "../target/release/infotheory"
      ITE.ensureExecutable rustBin
      ITE.runSequiturSuite rustBin
    else
      ITE.runSuite
  if ok then
    IO.println "[OK] ite-bench validation suite passed"
    return 0
  else
    IO.println "[FAIL] ite-bench validation suite failed"
    return 2
