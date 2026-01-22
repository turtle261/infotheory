import Std
import ITE.Types

namespace ITE

open Std

/-- Falsification rule thresholds from the specification. -/
structure FalsificationRule where
  threshold : Float
  description : String
  deriving Repr

structure FalsificationRules where
  metricViolation : FalsificationRule
  inequalityViolation : FalsificationRule
  oracleDivergenceAbs : Float
  oracleDivergenceRel : Float
  inconsistency : FalsificationRule
  boundViolation : FalsificationRule
  deriving Repr

/-- Default falsification rules mirroring the spec. -/
def defaultFalsificationRules : FalsificationRules :=
  { metricViolation := { threshold := 0.05, description := "Violates metric axioms in >5% of cases" }
  , inequalityViolation := { threshold := 0.01, description := "Violates fundamental inequalities" }
  , oracleDivergenceAbs := 0.5
  , oracleDivergenceRel := 0.30
  , inconsistency := { threshold := 0.20, description := "Relative stddev >20% on identical inputs" }
  , boundViolation := { threshold := 0.0, description := "Mathematical bound violations are never acceptable" }
  }

/-- Summary for metric property verification. -/
structure MetricSummary where
  nonNegativityPassRate : Float
  identityPassRate : Float
  symmetryPassRate : Float
  trianglePassRate : Float
  falsified : Bool
  warnings : Array String := #[]
  deriving Repr

/-- Estimator performance summary for a single quantity. -/
structure EstimatorPerformance where
  mae : Float := 0.0
  mape : Float := 0.0
  rmse : Float := 0.0
  maxError : Float := 0.0
  metricViolations : Float := 0.0
  inequalityViolations : Float := 0.0
  falsified : Bool := false
  deriving Repr

/-- Regime-stratified quantity report. -/
structure QuantityReport where
  oracleType : String
  rustCrate : EstimatorPerformance
  deriving Repr

/-- Regime-level report containing all quantities and metric validations. -/
structure RegimeReport where
  description : DataRegime
  quantities : Std.HashMap Quantity QuantityReport
  metricProperties : Std.HashMap Quantity MetricSummary
  deriving Repr

/-- Top-level benchmark report. -/
structure BenchmarkReport where
  benchmarkVersion : String
  date : String
  estimators : Array String
  regimes : Std.HashMap String RegimeReport
  falsifications : Array (Std.HashMap String String)
  deriving Repr

/-- Decide falsification based on counts and thresholds. -/
def decideFalsified (rules : FalsificationRules) (metricRate inequalityRate : Float)
    (mae mape : Float) (maxAbs maxRel : Float) : Bool :=
  metricRate > rules.metricViolation.threshold
    ∨ inequalityRate > rules.inequalityViolation.threshold
    ∨ maxAbs > rules.oracleDivergenceAbs
    ∨ maxRel > rules.oracleDivergenceRel

end ITE
