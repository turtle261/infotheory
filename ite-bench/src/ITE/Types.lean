import Std

namespace ITE

open Std

/-- Compact repr for raw bytes to avoid huge printouts. -/
instance : Repr ByteArray where
  reprPrec _ _ := "<bytes>"

/-- Quantities that estimators may produce. -/
inductive Quantity
  | shannonEntropy
  | klDivergence
  | mutualInformation
  | jointEntropy
  | conditionalEntropy
  | crossEntropy
  | jsDivergence
  | tvd
  | ncdVitanyi
  | ncdSymVitanyi
  | ncdCons
  | ncdSymCons
  | ned
  | nedCons
  | nedMI
  | nte
  | entropyRate
  deriving DecidableEq, Repr, Inhabited, Hashable

/-- Classification of alphabet/support size regimes. -/
inductive AlphabetSize
  | binary
  | small
  | medium
  | large
  | continuous
  deriving DecidableEq, Repr, Inhabited

/-- Classification of sample sizes. -/
inductive SampleSize
  | tiny
  | small
  | medium
  | large
  | asymptotic
  deriving DecidableEq, Repr, Inhabited

/-- Distribution families. -/
inductive DistributionFamily
  | uniform
  | highlySkewed
  | multimodal
  | heavyTailed
  | mixture
  | structured
  deriving DecidableEq, Repr, Inhabited

/-- Dependency strength classification. -/
inductive Dependence
  | independent
  | weak
  | moderate
  | strong
  | deterministic
  deriving DecidableEq, Repr, Inhabited

/-- Dimensionality regimes. -/
inductive Dimensionality
  | univariate
  | bivariate
  | lowDim
  | highDim
  deriving DecidableEq, Repr, Inhabited

/-- Compressibility regimes (for compression-based measures). -/
inductive Compressibility
  | incompressible
  | low
  | medium
  | high
  | structured
  deriving DecidableEq, Repr, Inhabited

/-- Stationarity regimes for rate-based measures. -/
inductive Stationarity
  | iid
  | markov
  | trending
  | switching
  deriving DecidableEq, Repr, Inhabited

/-- Core description of a data regime. Every test point belongs to exactly one regime. -/
structure DataRegime where
  sampleSize : SampleSize
  alphabetSize : AlphabetSize
  distribution : DistributionFamily
  dependence : Dependence
  dimensionality : Dimensionality
  compressibility : Compressibility
  stationarity : Stationarity
  deriving Repr, Inhabited

/-- Generic estimator parameters; kept agnostic to concrete implementations. -/
structure EstimatorParams where
  scalars : HashMap String Float := HashMap.empty
  deriving Repr, Inhabited

def defaultParams : EstimatorParams := { scalars := HashMap.empty }

/-- Data payload passed to estimators. Optional components let each estimator pick what it needs. -/
structure SampleBundle where
  x : Option (Array Float) := none
  y : Option (Array Float) := none
  z : Option (Array Float) := none
  xy : Option (Array (Float × Float)) := none
  bytesX : Option ByteArray := none
  bytesY : Option ByteArray := none
  bytesZ : Option ByteArray := none
  deriving Repr, Inhabited

/-- Unified estimator interface. -/
structure Estimator where
  name : String
  estimate : Quantity → SampleBundle → EstimatorParams → IO (Except String Float)

/-- Tolerances for metric axioms. -/
structure MetricTolerances where
  nonNegativity : Float
  identity : Float
  symmetry : Float
  triangle : Float
  deriving Repr

/-- Regime-dependent tolerance functions. -/
structure Tolerances where
  sampleSize : SampleSize → Float
  alphabetSize : AlphabetSize → Float
  metric : MetricTolerances
  quantity : Quantity → DataRegime → Float

namespace ToleranceDefaults

private def sampleSizeTol : SampleSize → Float
  | .tiny => 0.5
  | .small => 0.2
  | .medium => 0.1
  | .large => 0.05
  | .asymptotic => 0.01

private def alphabetTol : AlphabetSize → Float
  | .binary => 0.01
  | .small => 0.05
  | .medium => 0.1
  | .large => 0.2
  | .continuous => 0.3

private def metricTol : MetricTolerances :=
  { nonNegativity := 1e-10
  , identity := 1e-4
  , symmetry := 1e-6
  , triangle := 0.05
  }

/-- Quantity-specific tolerance selection following the spec. -/
private def quantityTol (q : Quantity) (r : DataRegime) : Float :=
  match q with
  | .shannonEntropy => sampleSizeTol r.sampleSize
  | .klDivergence => 2.0 * sampleSizeTol r.sampleSize
  | .mutualInformation => sampleSizeTol r.sampleSize
  | .jointEntropy => sampleSizeTol r.sampleSize
  | .conditionalEntropy => sampleSizeTol r.sampleSize
  | .crossEntropy => sampleSizeTol r.sampleSize
  | .jsDivergence => sampleSizeTol r.sampleSize
  | .tvd => 0.0 -- metric, not oracle error; violations handled elsewhere
  | .ncdVitanyi | .ncdSymVitanyi | .ncdCons | .ncdSymCons =>
      if r.compressibility = .high ∨ r.compressibility = .structured then 0.1 else 0.2
  | .ned | .nedCons | .nedMI => sampleSizeTol r.sampleSize
  | .nte => sampleSizeTol r.sampleSize
  | .entropyRate =>
      match r.stationarity with
      | .iid => 0.05
      | .markov => 0.05
      | .trending => 0.1
      | .switching => 0.1

/-- Default tolerance table mirroring the benchmark specification. -/
def defaults : Tolerances :=
  { sampleSize := sampleSizeTol
  , alphabetSize := alphabetTol
  , metric := metricTol
  , quantity := quantityTol
  }

end ToleranceDefaults

/-- Probability inequality identifiers for reporting. -/
inductive InequalityId
  | subadditivity
  | miNonnegative
  | dataProcessing
  | pinsker
  | fano
  deriving DecidableEq, Repr, Inhabited

/-- Result of a single property check. -/
structure CheckResult where
  passed : Bool
  value : Float
  oracle : Option Float := none
  note : Option String := none
  deriving Repr

end ITE
