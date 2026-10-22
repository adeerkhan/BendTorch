# performance/template_value_field: a field of a record built in a definition that takes templates (`~l`), whose value
# is not a lambda and reads one of those templates. A template is substituted at each use, so a record built from
# templates is built again wherever it is used; a lambda field costs a closure, but a value field is computed then, and
# reading a template there builds that template's whole value too, to throw most of it away.
# Only records with a lambda field are judged: a record of functions is what a program passes as a template; a record
# of values (a result, a state) is built once where it is computed.
# Heuristics: found in BendTorch's M9, where `count: Layer.count(la) + Layer.count(lb)` in the combinators made a
# two-layer model run 5.2% more instructions (term_drop 0.5% -> 4.7% of them) and 6% more time, the same output; as a
# closure (`count: u => ...`) the gap was 0.13%. The cost grows with how deep the templates nest. A template of a sort
# (`~P: Type`) or of a number, flag or string is cheap to read and not reported; nor is a field read under a lambda, nor
# a record built inside a lambda (built on demand).
import Base
import bend-lint-api@0.1.0.0/linter/lint.bend as Lint
import bend-lint-api@0.1.0.0/shared/lists.bend as Lists

# The binders of a definition's template parameters that the rule watches.
type Owner is Data:
  Owner{templates: List<&2, U32>}

# Whether a parameter whose type is `kids` carries a value worth computing: not a sort, a number, a flag or a string.
def heavy(kids: List<&2, Lint.Node>) -> Bool:
  match kids:
    case Con{Lint.Node{Lint.Sort{}, n, s, k, f}, rest}:
      False{}
    case Con{Lint.Node{Lint.Reference{t}, n, s, k, f}, rest}:
      Bool.not(List.contains(~String, ~String.eq, ["Nat", "U32", "I32", "F32", "Bool", "Char", "String", "Unit"], t))
    case Con{node, rest}:
      True{}
    case Nil{}:
      False{}

# The binders of the watched template parameters among a definition's nodes, onto `out`.
def templates(nodes: List<&2, Lint.Node>, out: List<&2, U32>) -> List<&2, U32>:
  match nodes:
    case Nil{}:
      out
    case Con{Lint.Node{Lint.Parameter{b, Lint.Template{}}, n, s, kids, f}, rest}:
      templates(rest, Lists.push(U32, Lists.when(U32, heavy(kids), b), out))
    case Con{node, rest}:
      templates(rest, out)

# Whether `nodes`, outside lambdas, use one of `ts`, or `found`.
def reads(nodes: List<&2, Lint.Node>, +ts: List<&2, U32>, found: Bool) -> Bool:
  match nodes found:
    case _ True{}:
      True{}
    case Nil{} False{}:
      False{}
    case Con{Lint.Node{Lint.Lambda{b, q}, n, s, k, f}, rest} False{}:
      reads(rest, ts, False{})
    case Con{Lint.Node{Lint.Variable{b}, n, s, k, f}, rest} False{}:
      reads(rest, ts, List.contains(~U32, ~U32.is_eq, ts, b))
    case Con{Lint.Node{syntax, n, s, kids, f}, rest} False{}:
      reads(rest, ts, reads(kids, ts, False{}))

# Whether a node below `path` is under a lambda.
def in_lambda(path: List<&2, Lint.Above>) -> Bool:
  match path:
    case Nil{}:
      False{}
    case Con{Lint.Above{Lint.Node{Lint.Lambda{b, q}, n, s, k, f}, at}, rest}:
      True{}
    case Con{a, rest}:
      in_lambda(rest)

def message() -> String:
  "this field is not a lambda and reads a template: the record is built again at every use of the template, and this "
    ++ "value is computed each time, building the template's own value to read it. In a combinator that composes "
    ++ "templates, every use of the composite then builds its children's whole records and drops them; the cost grows "
    ++ "with how deep the templates nest. Measured: 5.2% more instructions in a two-layer model, 0.13% with the field "
    ++ "as a closure. Make the field a lambda (`u => ...`, typed `Unit -> T`) and call it where the value is needed. "
    ++ "Confidence: medium; if the record is used only a few times, expect this finding."

# A finding for each field of a construction that is not a lambda and reads one of `ts`, onto `out`.
def fields(kids: List<&2, Lint.Node>, +ts: List<&2, U32>, out: List<&2, Lint.Finding>) -> List<&2, Lint.Finding>:
  match kids:
    case Nil{}:
      out
    case Con{Lint.Node{Lint.Lambda{b, q}, n, s, k, f}, rest}:
      fields(rest, ts, out)
    case Con{Lint.Node{+syntax, +n, +s, +k, +f}, rest}:
      fields(
        rest,
        ts,
        Lists.onto(
          &2,
          Lint.Finding,
          Lint.finding(Lint.Warning{}, reads([Lint.Node{syntax, n, s, k, f}], ts, False{}), s, message()),
          out
        )
      )

# Whether one of `kids` is a lambda: the record is one of functions, the kind a program passes as a template.
def has_lambda(kids: List<&2, Lint.Node>) -> Bool:
  match kids:
    case Nil{}:
      False{}
    case Con{Lint.Node{Lint.Lambda{b, q}, n, s, k, f}, rest}:
      True{}
    case Con{node, rest}:
      has_lambda(rest)

def built(hit: Bool, kids: List<&2, Lint.Node>, +ts: List<&2, U32>) -> List<&2, Lint.Finding>:
  match hit:
    case True{}:
      fields(kids, ts, [])
    case False{}:
      []

def visit(+owner: Owner, +path: List<&2, Lint.Above>, +node: Lint.Node) -> List<&2, Lint.Finding>:
  match owner node:
    case Owner{ts} Lint.Node{Lint.Construction{t}, n, s, +kids, f}:
      built(Lint.is_live(path) && Bool.not(in_lambda(path)) && has_lambda(kids), kids, ts)
    case _ _:
      []

def defs(nodes: List<&2, Lint.Node>, out: List<&2, Lint.Finding>) -> List<&2, Lint.Finding>:
  match nodes:
    case Nil{}:
      List.reverse(&2, Lint.Finding, out)
    case Con{Lint.Node{Lint.Definition{t, u}, n, s, +k, f}, rest}:
      defs(
        rest,
        Lists.onto(
          &2,
          Lint.Finding,
          Lint.visit(~Owner, ~Lint.Finding, ~visit, Owner{templates(k, [])}, [Lint.Node{Lint.Definition{t, u}, n, s, k, f}]),
          out
        )
      )
    case Con{node, rest}:
      defs(rest, out)

def run(+here: Lint.Here) -> Lint.Ask(List<&2, Lint.Finding>):
  Lint.done(defs(Lint.Here.nodes(here), []))

def rule() -> Lint.Rule:
  Lint.Rule{Lint.Needs{False{}, False{}}, [], Lint.OnFile{run}}
