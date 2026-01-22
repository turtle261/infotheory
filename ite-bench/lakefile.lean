import Lake
open Lake DSL

package «ite-bench» where
  srcDir := "src"
  moreServerArgs := #[]
  moreLeanArgs := #[]

@[default_target]
lean_lib ITE where
  globs := #[.submodules `ITE]

lean_exe itebench where
  root := `Main
  supportInterpreter := true

lean_exe runner where
  root := `Runner
  supportInterpreter := true
