import Std
import ITE.Types

namespace ITE

open Std
open IO
open System

/-- Basic float parser for decimal strings with optional sign. This avoids unavailable std helpers. -/
def parseFloatSimple (s : String) : Except String Float :=
  let t := s.trim
  -- Handle special float values
  if t == "nan" || t == "-nan" || t == "NaN" then
    .error "Estimator returned NaN"
  else if t == "inf" || t == "Inf" then
    .error "Estimator returned positive infinity"
  else if t == "-inf" || t == "-Inf" then
    .error "Estimator returned negative infinity"
  else
    let (sign, body) :=
      if t.startsWith "-" then (-1.0, t.drop 1) else if t.startsWith "+" then (1.0, t.drop 1) else (1.0, t)
    let parts := body.splitOn "."
    let fromNat (n : Nat) : Float := Float.ofNat n
    match parts with
    | [intPart] =>
      match intPart.toNat? with
      | some n => .ok <| sign * fromNat n
      | none => .error s!"Could not parse float from: {s}"
    | [intPart, fracPart] =>
      match intPart.toNat?, fracPart.toNat? with
      | some n, some f =>
        let denom := fromNat (Nat.pow 10 fracPart.length)
        let val := fromNat n + (fromNat f / denom)
        .ok <| sign * val
      | _, _ => .error s!"Could not parse float from: {s}"
    | _ => .error s!"Could not parse float from: {s}"



/-- Write a byte array to a temporary file and return its path. -/
def writeTempBytes (stem : String) (bytes : ByteArray) : IO FilePath := do
  let base := FilePath.mk "/tmp/ite-bench"
  IO.FS.createDirAll base
  let nonce ← IO.rand 0 1000000000
  let path := base / s!"{stem}_{nonce}.bin"
  IO.FS.writeBinFile path bytes
  pure path

/-- Convert an array of `Float` samples (interpreted as discrete symbols 0-255) into bytes. -/
def floatsToBytes (xs : Array Float) : ByteArray :=
  xs.foldl (init := ByteArray.empty) (fun acc v =>
    let clipped := if v < 0.0 then 0.0 else if v > 255.0 then 255.0 else v
    acc.push (UInt8.ofNat <| clipped.toUInt64.toNat))

/-- Materialize `SampleBundle` components to byte file paths for CLI estimators. -/
structure ByteInputs where
  xPath : Option FilePath := none
  yPath : Option FilePath := none
  zPath : Option FilePath := none
  deriving Inhabited

/-- Convert Float arrays to a simple 2D row-vector representation (rows = samples). -/
def floatsTo2D (xs : Array Float) : List (List Float) :=
  xs.toList.map (fun v => [v])

/-- Convert paired Float arrays into list-of-pairs rows. -/
def floatsPairTo2D (xy : Array (Float × Float)) : List (List Float) :=
  xy.toList.map (fun (a, b) => [a, b])

/-- Build byte-backed inputs from a `SampleBundle`, converting arrays when needed. -/
def bundleToByteInputs (b : SampleBundle) : IO (Except String ByteInputs) := do
  let mut res : ByteInputs := {}
  match b.bytesX, b.x with
  | some bx, _ => res := { res with xPath := some <| ← writeTempBytes "x" bx }
  | none, some xarr => res := { res with xPath := some <| ← writeTempBytes "x" (floatsToBytes xarr) }
  | none, none => pure ()
  match b.bytesY, b.y with
  | some yb, _ => res := { res with yPath := some <| ← writeTempBytes "y" yb }
  | none, some yarr => res := { res with yPath := some <| ← writeTempBytes "y" (floatsToBytes yarr) }
  | none, none => pure ()
  match b.bytesZ, b.z with
  | some bz, _ => res := { res with zPath := some <| ← writeTempBytes "z" bz }
  | none, some zarr => res := { res with zPath := some <| ← writeTempBytes "z" (floatsToBytes zarr) }
  | none, none => pure ()
  return .ok res

/-- Map `Quantity` to the infotheory CLI primitive and arity. -/
inductive PrimitiveArity
  | unary (name : String)
  | binary (name : String)
  deriving Repr

private def infotheoryPrimitive (q : Quantity) : Option PrimitiveArity :=
  match q with
  | .shannonEntropy => some <| .unary "entropy"
  | .entropyRate => some <| .unary "entropy_rate"
  | .mutualInformation => some <| .binary "mi"
  | .jointEntropy => some <| .binary "joint_entropy"
  | .conditionalEntropy => some <| .binary "ce"
  | .crossEntropy => some <| .binary "xe"
  | .klDivergence => some <| .binary "kl"
  | .jsDivergence => some <| .binary "js"
  | .ncdVitanyi => some <| .binary "ncd_vitanyi"
  | .ncdSymVitanyi => some <| .binary "ncd_sym_vitanyi"
  | .ncdCons => some <| .binary "ncd_cons"
  | .ncdSymCons => some <| .binary "ncd_sym_cons"
  | .ned => some <| .binary "ned"
  | .nedCons => some <| .binary "ned_cons"
  | .nedMI => some <| .binary "ned"
  | .nte => some <| .binary "nte"
  | .tvd => some <| .binary "tvd"

/-- Adapter for the Rust `infotheory` CLI.

Default path assumes this Lean project is run from `ite-bench/` and the Rust
workspace is built at the repository root.
-/
def infotheoryEstimator (binPath : FilePath := FilePath.mk "../target/release/infotheory") : Estimator :=
  { name := s!"infotheory({binPath})"
  , estimate := fun q bundle _params => do
      match infotheoryPrimitive q with
      | none => return .error s!"Quantity {repr q} not supported by infotheory CLI"
      | some prim =>
        match ← bundleToByteInputs bundle with
        | .error e => return .error e
        | .ok bytes =>
          let params := _params
          let paramString (k : String) : Option String :=
            params.strings[k]?
          let rateBackendStr := paramString "rate_backend"
          let compressionBackendStr := paramString "compression_backend"
          let methodStr := paramString "method"

          let withCommonFlags (args : Array String) : Array String :=
            let args := match rateBackendStr with
              | some rb => args.push "--rate-backend" |>.push rb
              | none => args
            let args := match compressionBackendStr with
              | some nb => args.push "--compression-backend" |>.push nb
              | none => args
            let args := match methodStr with
              | some m => args.push "--method" |>.push m
              | none => args
            args

          let runUnary (primName : String) (path : FilePath) : IO (Except String Float) := do
            let args := withCommonFlags #[primName, path.toString]
            let out ← IO.Process.output { cmd := binPath.toString, args := args }
            if out.exitCode ≠ 0 then
              return .error s!"infotheory call failed: {out.stderr}"
            return parseFloatSimple out.stdout

          let runBinary (primName : String) (p1 p2 : FilePath) : IO (Except String Float) := do
            let args := withCommonFlags #[primName, p1.toString, p2.toString]
            let out ← IO.Process.output { cmd := binPath.toString, args := args }
            if out.exitCode ≠ 0 then
              return .error s!"infotheory call failed: {out.stderr}"
            return parseFloatSimple out.stdout

          match prim with
          | .unary name =>
            match bytes.xPath with
            | some px => runUnary name px
            | none => pure <| .error "infotheory unary requires X bytes"
          | .binary name =>
            match (bytes.xPath, bytes.yPath) with
            | (some px, some py) => runBinary name px py
            | _ => pure <| .error "infotheory binary requires X and Y bytes"
  }

end ITE
