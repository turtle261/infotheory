import Std
import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace ITE

open Std
open IO
open Lean
open System

structure SequiturDebugRule where
  id : Nat
  rhs : Array Int
  deriving Repr, Inhabited, FromJson, ToJson

structure SequiturDebugCase where
  input_hex : String
  decoded_hex : String
  rules : Array SequiturDebugRule
  trace : Array (Array Float)
  deriving Repr, Inhabited, FromJson, ToJson

structure SequiturDebugBatch where
  context_bytes : Nat
  alphabet_prefix : Nat
  cases : Array SequiturDebugCase
  deriving Repr, Inhabited, FromJson, ToJson

structure DigramOcc where
  ruleId : Nat
  pos : Nat
  left : Int
  right : Int
  deriving Repr, Inhabited, BEq, DecidableEq

private def arrayAllIdx {α : Type} [Inhabited α] (xs : Array α) (f : Nat → α → Bool) : Bool :=
  Id.run do
    let mut ok := true
    for idx in [:xs.size] do
      if !f idx (xs.get! idx) then
        ok := false
    ok

private def byteArrayEq (a b : ByteArray) : Bool :=
  if a.size != b.size then
    false
  else
    Id.run do
      let mut ok := true
      for idx in [:a.size] do
        if a.get! idx != b.get! idx then
          ok := false
      ok

private def hexNibble? (c : Char) : Option UInt8 :=
  if '0' ≤ c ∧ c ≤ '9' then
    some <| UInt8.ofNat (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then
    some <| UInt8.ofNat (10 + c.toNat - 'a'.toNat)
  else if 'A' ≤ c ∧ c ≤ 'F' then
    some <| UInt8.ofNat (10 + c.toNat - 'A'.toNat)
  else
    none

def hexToBytes (raw : String) : Except String ByteArray := do
  let chars := raw.data.filter fun c => !c.isWhitespace && c ≠ '_'
  if chars.length % 2 != 0 then
    throw "hex input must have an even number of digits"
  let mut out := ByteArray.empty
  let mut i := 0
  while i < chars.length do
    let hi := chars.get! i
    let lo := chars.get! (i + 1)
    let some hiNib := hexNibble? hi
      | throw s!"invalid hex digit '{hi}'"
    let some loNib := hexNibble? lo
      | throw s!"invalid hex digit '{lo}'"
    out := out.push <| UInt8.ofNat (16 * hiNib.toNat + loNib.toNat)
    i := i + 2
  pure out

def bytesToHex (bytes : ByteArray) : String :=
  let hex := "0123456789abcdef".data.toArray
  Id.run do
    let mut out := ""
    for idx in [:bytes.size] do
      let byte := bytes.get! idx
      out := out.push (hex.get! (byte.toNat / 16))
      out := out.push (hex.get! (byte.toNat % 16))
    out

private def decodeSym? (sym : Int) : Except String (Sum UInt8 Nat) :=
  if sym >= 0 then
    if sym > 255 then
      throw s!"terminal out of range: {sym}"
    else
      pure <| .inl <| UInt8.ofNat sym.natAbs
  else
    pure <| .inr (Int.natAbs (-sym - 1))

private partial def decodeRuleWithFuel
    (rules : Array SequiturDebugRule) (fuel : Nat) (ruleId : Nat) : Except String ByteArray := do
  if fuel == 0 then
    throw "decode fuel exhausted (possible nonterminal cycle)"
  let some rule := rules.get? ruleId
    | throw s!"missing rule {ruleId}"
  if rule.id != ruleId then
    throw s!"rule id/index mismatch: expected {ruleId}, got {rule.id}"
  let mut out := ByteArray.empty
  for sym in rule.rhs do
    match ← decodeSym? sym with
    | .inl byte => out := out.push byte
    | .inr child =>
      let childBytes ← decodeRuleWithFuel rules (fuel - 1) child
      out := out ++ childBytes
  pure out

def decodeRoot? (rules : Array SequiturDebugRule) : Except String ByteArray :=
  decodeRuleWithFuel rules (rules.size.succ * 1024) 0

private def ruleUtilities (rules : Array SequiturDebugRule) : Array Nat :=
  Id.run do
    let mut counts := mkArray rules.size 0
    for rule in rules do
      for sym in rule.rhs do
        if sym < 0 then
          let child := Int.natAbs (-sym - 1)
          if h : child < counts.size then
            counts := counts.set ⟨child, h⟩ (counts[child]! + 1)
    counts

private def ruleBodiesWellShaped (rules : Array SequiturDebugRule) : Bool :=
  arrayAllIdx rules fun idx rule =>
    rule.id == idx && (idx == 0 || rule.rhs.size >= 2)

private def digramOccurrences (rules : Array SequiturDebugRule) : Array DigramOcc :=
  Id.run do
    let mut out := #[]
    for rule in rules do
      if rule.rhs.size >= 2 then
        for pos in [:rule.rhs.size - 1] do
          out := out.push {
            ruleId := rule.id
            pos := pos
            left := rule.rhs.get! pos
            right := rule.rhs.get! (pos + 1)
          }
    out

private def overlap (a b : DigramOcc) : Bool :=
  a.ruleId == b.ruleId && (a.pos + 1 == b.pos || b.pos + 1 == a.pos)

private def digramUnique (rules : Array SequiturDebugRule) : Bool :=
  let occs := digramOccurrences rules
  arrayAllIdx occs fun idx occ =>
    arrayAllIdx occs fun jdx other =>
      idx == jdx || occ.left != other.left || occ.right != other.right || overlap occ other

private def traceLooksSane (alphabetPrefix : Nat) (trace : Array (Array Float)) : Bool :=
  trace.all fun row =>
    row.size == alphabetPrefix &&
      row.all fun p =>
        !p.isNaN && p > 0.0 && p < 1.0

def validateDebugCase (batch : SequiturDebugBatch) (case : SequiturDebugCase) : Except String Unit := do
  let inputBytes ← hexToBytes case.input_hex
  let decodedBytes ← hexToBytes case.decoded_hex
  if !byteArrayEq decodedBytes inputBytes then
    throw s!"decoded_hex mismatch for input {case.input_hex}"
  let decodedFromRules ← decodeRoot? case.rules
  if !byteArrayEq decodedFromRules inputBytes then
    throw s!"grammar decode mismatch for input {case.input_hex}"
  if !ruleBodiesWellShaped case.rules then
    throw s!"rule shape invariant failed for input {case.input_hex}"
  let utilities := ruleUtilities case.rules
  if !(arrayAllIdx utilities fun idx count => idx == 0 || count >= 2) then
    throw s!"rule utility invariant failed for input {case.input_hex}"
  if !digramUnique case.rules then
    throw s!"digram uniqueness failed for input {case.input_hex}"
  if !traceLooksSane batch.alphabet_prefix case.trace then
    throw s!"trace shape/probability sanity failed for input {case.input_hex}"

def validateDebugBatch (batch : SequiturDebugBatch) : Except String Unit := do
  for case in batch.cases do
    validateDebugCase batch case

def parseDebugBatch (raw : String) : Except String SequiturDebugBatch := do
  let json ← Json.parse raw
  fromJson? json

def allWordsUpTo (alphabet : Array UInt8) (maxLen : Nat) : Array ByteArray :=
  let rec exactWords (len : Nat) : Array ByteArray :=
    match len with
    | 0 => #[ByteArray.empty]
    | n + 1 =>
      let shorter := exactWords n
      Id.run do
        let mut out : Array ByteArray := #[]
        for pre in shorter do
          for byte in alphabet do
            out := out.push (pre.push byte)
        out
  Id.run do
    let mut out : Array ByteArray := #[]
    for len in List.range (maxLen + 1) do
      out := out ++ exactWords len
    out

def batchHexInputs (inputs : Array ByteArray) (chunkSize : Nat) : Array (Array String) :=
  if chunkSize == 0 then
    #[inputs.map bytesToHex]
  else
    let hexes := inputs.map bytesToHex
    Id.run do
      let mut start := 0
      let mut acc : Array (Array String) := #[]
      while start < hexes.size do
        let stop := Nat.min (start + chunkSize) hexes.size
        let chunk := hexes.extract start stop
        acc := acc.push chunk
        start := stop
      acc

def runRustSequiturDebug
    (binPath : System.FilePath)
    (hexInputs : Array String)
    (contextBytes : Nat := 64)
    (alphabetPrefix : Nat := 4) : IO (Except String SequiturDebugBatch) := do
  let mut args : Array String := #[
    "sequitur-debug",
    "--context-bytes", toString contextBytes,
    "--alphabet-prefix", toString alphabetPrefix
  ]
  for hex in hexInputs do
    args := args.push "--hex"
    args := args.push hex
  let out ← IO.Process.output { cmd := binPath.toString, args := args }
  if out.exitCode != 0 then
    return .error s!"sequitur-debug failed: {out.stderr}"
  pure <| parseDebugBatch out.stdout

end ITE
