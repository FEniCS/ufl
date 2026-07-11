# UFL

UFL (Unified Form Language) is a domain-specific language, embedded in Python, for declaring finite
element variational forms and the function spaces they are built on. It is part of the FEniCS Project
and is the shared symbolic layer consumed by multiple form compilers and simulation packages —
FFCx, TSFC (used by Firedrake), and DOLFINx — none of which re-implement form algebra or
differentiation themselves; they all lower whatever `compute_form_data` hands back.

## Project Architecture

* **Two separate class hierarchies, not one.** `Expr` (`ufl/core/expr.py`) is the *primal* algebra:
  scalars, tensors, `Coefficient`, `Argument`, and everything built from them with `+`, `*`, `grad`,
  etc. `BaseForm` (`ufl/form.py`) is the *dual*, form-level algebra: `Form` (a sum of `Integral`s),
  `FormSum`, `Matrix`, `Cofunction`, `Coargument`, `Action`, `Adjoint`, `ZeroBaseForm`. A `BaseForm` is
  not an `Expr` and vice versa — code that dispatches on `Expr` (e.g. `isinstance(x, Expr)`, or a
  `singledispatchmethod` registered on `Expr`) will silently miss every `BaseForm` node, which is a
  common source of "no rule found" failures when a new `BaseForm` subtype is introduced somewhere in a
  pipeline that was only ever exercised with plain `Form`s.
* **`compute_form_data` is the single entry point form compilers rely on** (`ufl/algorithms/compute_form_data.py`).
  It runs, in order: derivative expansion (`apply_derivatives`), pullback/geometry lowering, default
  restrictions, and integral scaling, converging on a small, uniform vocabulary of node types so that
  FFCx/TSFC do not each need to understand the full UFL language.
* **Differentiation is a `singledispatchmethod` dispatch tree**, not a single function. `apply_derivatives.py`
  defines a hierarchy of `DAGTraverser` subclasses (`GenericDerivativeRuleset` → `GateauxDerivativeRuleset`
  → `BaseFormOperatorDerivativeRuleset`, plus the top-level `DerivativeRuleDispatcher`), each registering
  one `process.register(SomeType)` handler per node type it knows how to differentiate. Because of a
  known CPython limitation with `singledispatchmethod` across inheritance
  (https://bugs.python.org/issue36457), every subclass re-declares its own `process` singledispatchmethod
  and falls back to `super().process(o)` for anything it does not override itself — a rule registered on
  a parent class *is* reachable from a subclass instance, but only through that explicit fallback chain,
  never through ordinary Python MRO. When adding a rule for a new type, find the *nearest* existing rule
  for a structurally similar type (e.g. `Matrix`/`Cofunction`/`Coargument` all being "terminal-like dual
  objects with no dependence on a plain `Coefficient`") and add the sibling rule to the same class.
* **`map_integrands` (`ufl/algorithms/map_integrands.py`) is the generic recursive form transform.**
  Nearly every form-rewriting algorithm (differentiation, replacement, splitting, action/adjoint
  construction) is implemented by handing a per-`Expr` callable to `map_integrands`, which recurses
  through `Form`/`Integral`/`FormSum`/`Action`/`Adjoint`/`ZeroBaseForm`/plain `Expr`, applying the
  callable to integrands and recombining the result. Its branches are not symmetric: the `FormSum`
  branch can (and must) derive a fully-cancelled result's argument shape from an already-mapped
  component, but the plain `Form` branch has no such component to draw from — see the Anti-Patterns
  section below before "fixing" one to mirror the other.
* **`ZeroBaseForm` is the dual-space zero, and it carries shape.** Unlike `ufl.Zero` (a primal,
  shapeless-by-comparison placeholder) or a bare Python `0`, a `ZeroBaseForm` stores the `Argument`s
  (and, transiently, the direction of whatever differentiation produced it) that the now-vanished
  quantity would have depended on. Losing that information — by collapsing to `Form([])` or `0` instead
  — is invisible until something downstream calls `.arguments()` on the result and gets the wrong
  answer or an empty tuple.

## Core Working Rules

* **Mathematical Root Causes:** Fix the underlying rule gap or structural mismatch, not the individual
  failing test. If a crash traces back to a missing `process.register` case, add the rule; do not catch
  the exception at the call site.
* **Generality Over Complexity:** UFL's value is that a small set of orthogonal node types compose
  freely. Prefer extending the dispatch tables that already exist (`GenericDerivativeRuleset`,
  `map_integrands`, `extract_type`) over adding a special case in a specific algorithm.
* **Preserve Style:** Match the surrounding module's docstring style (Google-style `Args:`/`Returns:`,
  not numpydoc) and keep edits minimal and local to the requested change.
* **Do Not Trust Memorized API Shapes:** The primal/dual (`Expr`/`BaseForm`) split, `Cofunction`,
  `Coargument`, `Interpolate`, and `ZeroBaseForm` are all comparatively recent additions layered onto an
  older, `Expr`-only mental model. Code, docs, or trained knowledge describing "a form is a sum of
  integrals over `Expr`s" predates this split and will be wrong about how duals, `Action`, and `Adjoint`
  behave. Read the actual class (`ufl/form.py`, `ufl/action.py`, `ufl/adjoint.py`) before assuming a
  method exists or behaves a particular way.
* **Document The Present, Not The Past:** Do not leave comments explaining what an old, now-removed
  code path used to do. If a previous workaround is now unnecessary because a root cause was fixed
  elsewhere, say what is true now and why, not what used to be broken.

## Coding Style And Conventions

* **Type Hints And Docstrings:** New code should include type hints on function/method signatures and
  Google-style (`Args:`, `Returns:`) docstrings, matching the rest of `ufl/algorithms/`.
* **`# type: ignore` on AD rule registrations:** A `process.register(SomeType)` handler whose signature
  narrows the base `process(self, o: Expr) -> Expr` (e.g. to `(self, o: SomeType) -> BaseForm`) will
  fail `mypy` as an incompatible callable even though the runtime dispatch is correct. Add
  `# type: ignore` on the `@process.register(...)` line, matching the existing convention on the
  `Matrix`/`Interpolate`/`ExternalOperator` rules in `apply_derivatives.py`, rather than widening the
  type hints to something less precise.
* **`ruff` line length is 100**, not the flake8 default of 79 — `pyproject.toml`'s `[tool.ruff]` is
  authoritative; do not "fix" lines that are within 100 characters based on a bare `flake8` run.

## Pattern Matching For Planning And Debugging

* **Borrow a design from one layer down the stack before inventing one.** UFL sits below nothing and
  above several form compilers (FFCx, TSFC/GEM) that lower whatever `compute_form_data` produces. A
  simplification problem at the UFL layer has often already been solved, in a structurally analogous
  form, one layer further down — GEM already recognizes and cancels Kronecker-delta-producing index
  contractions during its own optimization passes. `cancel_jacobian_products.py`'s
  `JacobianCanceller`/`IdentityEliminator` pair (rewrite a contraction into an indexed `Identity`, then
  eliminate the `Identity` by substitution) is a deliberate port of that GEM pass, just run earlier —
  before pullback lowering destroys the structure GEM would otherwise have to rediscover on its own.
  Before designing a new algorithm for "cancel this recurring pattern", grep the downstream compiler for
  a pass that already names it; porting a design that has already shipped and been tested is strictly
  better than re-deriving one, and its existing tests hand you the edge cases for free. Treat this kind
  of borrowing across the stack as good practice, not a shortcut to apologize for.
* **Inside UFL, find the nearest *structurally* similar rule, not just the nearest class in the
  hierarchy.** A new node type needing a dispatch rule should be matched to an existing rule for a type
  playing the same abstract role (terminal-like dual object, zero-producing simplification,
  index-contraction rewrite) even if it lives in an unrelated class hierarchy. Copy the boilerplate that
  comes with it too — the `super().process(o)` fallback redeclaration, the `# type: ignore` on a narrowed
  `process.register` signature — it exists because of concrete constraints (Python class-body name
  resolution, `mypy`'s callable variance), not convention for its own sake, and skipping it reproduces
  failures someone already solved when they wrote the neighbor you copied.
* **Classify which structural category a fix belongs to before copying it to a sibling code path.** Two
  branches that look parallel are not automatically the same problem: one may only ever *add* information
  to a result, where its sibling *replaces* it (see the `map_integrands` `Form`-vs-`FormSum` case in
  Anti-Patterns below). State in one sentence which category the code you are changing falls into before
  generalizing a fix to a branch you have not actually exercised — a full local test-suite pass is not
  proof the generalization is safe; it may take a downstream consumer's own test suite to reveal the
  mismatch.
* **When a generalization regresses something only a downstream consumer's tests catch, isolate with a
  one-delta script before re-reading the whole diff.** Build the smallest example that drives exactly the
  two competing code paths (e.g. a node's pre-transform vs. post-transform arguments) through the
  algorithm directly, and `repr()` the result at each step — this localizes which rule produced the wrong
  answer far faster than reasoning about the full pipeline in the abstract.

## Testing Requirements

* **Pull Requests:** All PRs must include tests demonstrating the fix or feature.
* **Keep tests targeted.** Add or extend a test in the existing file that already covers the
  feature/module (e.g. `test/test_duals.py` for `BaseForm`/`Action`/`Adjoint`/`ZeroBaseForm` behavior).
  Do not create new test files for a single fix.
* Tests build finite elements and domains via `test/utils.py`'s `LagrangeElement` helper rather than
  going through a form compiler — UFL's own test suite exercises the symbolic layer in isolation.

## Pull Request Expectations

* Only a small set of maintainers can push directly to `origin` (`FEniCS/ufl`). Contributors push
  branches to their own fork remote and open the PR from `<fork-org>:<branch>` against
  `FEniCS/ufl:main`:
  ```bash
  git checkout -b <branch> main   # branch off an up-to-date origin/main
  # ... commit ...
  git push -u fork <branch>
  gh pr create --repo FEniCS/ufl --base main --head <fork-org>:<branch> --title ... --body ...
  ```
  Pushing to `origin` will fail with a permission error; that is expected, not a sign something is
  misconfigured — push to `fork` instead.
* Before opening or updating a PR, run all three lint stages `.github/workflows/lint.yml` runs, not
  just one of them:
  ```bash
  ruff check .
  ruff format --check .
  mypy -p ufl && mypy test/ && (cd test && mypy ../demo/)
  ```
  `ruff check .` passing locally is not sufficient evidence that CI's `lint` job will pass — the same
  job also runs `ruff format --check` and `mypy`, and a change that is algorithmically correct and
  ruff-clean can still fail `mypy` (e.g. a new `process.register` handler needing `# type: ignore`).

## Development Toolchain

### Environment Setup

* **Branch pairing with downstream consumers:** UFL's `main` pairs with the `main` branches of FFCx,
  TSFC/Firedrake, and DOLFINx; `release` pairs with their `release` branches. When co-developing a fix
  that touches both UFL and a downstream consumer (as this file's own introduction was), install UFL in
  editable mode (`pip install -e .`) in the consumer's environment and check which branch is checked
  out in each component before assuming a failure belongs to the package you are currently editing.
* **`test/utils.py`** provides `LagrangeElement` and other lightweight element/domain constructors used
  throughout `test/` to build symbolic examples without a full form-compiler stack — reach for it first
  when writing a reproduction script instead of constructing `FiniteElement`/`FunctionSpace` by hand.

### Testing

* `pytest test/` runs the full suite (a few hundred tests, seconds, no MPI/parallel infrastructure
  unlike downstream consumers such as Firedrake).
* **Narrow reproduction first:** write a standalone script building the minimal symbolic example (a
  `FunctionSpace` on one or two `LagrangeElement`s, plus the specific `derivative`/`action`/`adjoint`
  call) before running the full suite; UFL's failures are almost always reproducible in under twenty
  lines since the whole layer is symbolic and has no external dependencies to configure.

### Debugging

* **`repr()`/`str()` a `BaseForm` before trusting `.arguments()` or `.empty()`.** Printing a form (or
  the intermediate result of a `map_integrands`/`expand_derivatives` call) shows exactly which node
  types survived — a stray, unexpanded `CoefficientDerivative` node inside what should be a plain `Form`
  is immediately visible in the `repr`, whereas `.arguments()` alone can look plausible while being
  wrong (see Anti-Patterns).
* **`.signature()` equality is the idiom for "these two forms are the same up to argument renumbering"**
  in tests (see `test/test_extract_blocks.py`, or `tests/firedrake/regression/test_split.py` downstream)
  — build the expected form by hand from the same primitives and compare signatures rather than
  reconstructing and comparing `repr()` strings, which are sensitive to incidental numbering.
* **A `singledispatchmethod` `AssertionError: Rule not set for <type>`** means exactly one thing: no
  `process.register` handler exists for that type on the class actually being dispatched through (or any
  of its `super().process(o)` fallbacks). Find the class via `type(traverser)` at the failure point, not
  by guessing from the traceback alone — the same node type can be legally unhandled in one rule-set
  class (e.g. `GenericDerivativeRuleset`, which is abstract about `FormArgument`) while needing a
  concrete rule in a subclass (`GateauxDerivativeRuleset`).

## Anti-Patterns

### Using A Bare `0` Or `Form([])` For A Form That Turned Out To Be Zero

WRONG — Once an algorithm establishes that a form is identically zero, returning a bare Python `0` (or
an empty, argument-less `Form([])`) throws away which function spaces the zero result lived on:

```python
# Anti-pattern: the caller cannot recover which Arguments this was supposed to have
def compute_something(form):
    result = ...  # turns out every integral cancelled
    if not result:
        return Form([])
    return result
```

RIGHT — Construct a `ZeroBaseForm` carrying the arguments the (now-vanished) result would have had.
Downstream code that calls `.arguments()`, assembles into a specific block, or feeds the result into
further form algebra then keeps working exactly as it would for a nonzero result:

```python
def compute_something(form):
    result = ...
    if not result:
        return ZeroBaseForm(form.arguments())
    return result
```

This is precisely the shape of the bug that produced `AssertionError: Rule not set for ZeroBaseForm` in
`expand_derivatives`: a differentiation rule for `Matrix`/`Cofunction`/`Coargument` already built a
correctly-shaped `ZeroBaseForm`, but differentiating *that* `ZeroBaseForm` a second time (e.g. a Hessian)
hit a rule set with no registered handler for `ZeroBaseForm` itself — the fix was to add the missing
sibling rule, not to special-case the crash.

### Assuming A Transform's Post-Image Arguments Equal Its Pre-Image Arguments

WRONG — When a fully-cancelled `Form`'s zero result needs its arguments preserved, reaching for the
*original*, pre-transform form's `.arguments()` looks like the obvious fix, and is correct for
transforms that only ever *add* information (differentiation appends one new direction `Argument` on
top of the existing ones — the original arguments are still exactly right). It is wrong for transforms
that *replace* arguments, such as splitting a mixed-space form into a sub-block: the pre-transform form's
arguments are on the original, un-split function space, not the collapsed subspace the caller actually
wants.

```python
# Anti-pattern: correct for a pure differentiation pass, silently wrong for
# anything (like a mixed-space block splitter) that replaces Arguments
# rather than only adding new ones
if not nonzero_integrals:
    return ZeroBaseForm(form.arguments())  # `form` is the PRE-transform Form
```

RIGHT — A generic utility like `map_integrands` cannot know, for an arbitrary caller-supplied
transform, whether that transform only adds arguments or also replaces them; it should not guess. Let
the specific caller — which does know its own transform's semantics — reconstruct the correct arguments
itself, e.g. by re-applying its own per-`Argument` substitution rule to `form.arguments()`:

```python
# Correct: the caller (e.g. a mixed-space block splitter) knows how it
# maps an Argument onto the collapsed subspace, and rebuilds accordingly
f = map_integrand_dags(splitter, form)
if expand_derivatives(f).empty():
    f = ZeroBaseForm(tuple(map(splitter._subspace_argument, form.arguments())))
```

The `FormSum` branch of `map_integrands` gets to take the shortcut the `Form` branch cannot: when every
component of a `FormSum` vanishes, each component was *already* passed through the caller's transform
(`map_integrands` recurses into `FormSum.components()` before checking for cancellation), so
`mapped_components[0].arguments()` is the correctly-transformed shape, not a pre-image guess. Do not
generalize that shortcut to the plain `Form` branch, where no such already-mapped object exists.
