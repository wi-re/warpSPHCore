#!/usr/bin/env python3
"""Render the reviewed survey and source catalog from inventory.json."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--report-directory', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text())
    directory = args.report_directory
    directory.mkdir(parents=True, exist_ok=True)
    stem = 'native_operator_accumulators_2026_10_31'
    catalog_name = stem + '_catalog.md'
    roots = {name: Path(record['root']) for name, record in data['repositories'].items()}
    catalog_target = roots['core'] / 'docs/regression' / catalog_name
    inventory_target = roots['core'] / 'experiments/operator_survey/inventory.json'

    def source_link(package, path, line=None, label=None):
        target = str(roots[package] / path) + (f':{line}' if line else '')
        return f'[{label or path}]({target})'

    def anchor(package, path, name):
        return next(item for item in data['functions']
                    if item['package'] == package and item['path'] == path and item['name'] == name)

    def ref(package, path, name, label):
        item = anchor(package, path, name)
        return source_link(package, path, item['line'], label)

    counts = data['counts']
    categories = data['operation_categories']
    additive = categories['sum'] + categories['finalize']
    ratings = Counter(item['complexity'] for item in data['operations'])
    assert len(data['functions']) == 771 and len(data['files']) == 172
    assert len(data['operations']) == 113
    assert (ratings['low'],ratings['medium'],ratings['high']) == (43,48,22)
    assert data['core_reference']['different_definition_paths'] == [
        'src/warpSPHCore/math/wp_divdot.py',
        'src/warpSPHCore/math/wp_outerTensorProduct.py',
    ]
    size_summary = [
        ('Density', 'one scalar', '1', 'identity'),
        ('Scalar interpolation', 'one scalar', '1', 'identity'),
        ('Scalar gradient', 'one vector', '3', 'identity'),
        ('Vector Jacobian', 'one matrix', '9', 'identity'),
        ('Covariance', 'matrix and count', '9 floating + integer', 'identity'),
        ('CRK density', 'numerator and denominator', '2', 'guarded ratio'),
        ('CRK volume', 'kernel sum', '1', 'guarded reciprocal'),
        ('Psi and material psi', 'two scalar sums', '2', 'two dimension roots'),
        ('Incompressible alpha', 'vector sum and scalar sum', '4', 'squared norm and weighted sum'),
        ('mDBC shift', 'vector and per-axis counts', '3 floating + 3 integers', 'componentwise ratios'),
        ('CRK moments', 'six tensors and count', '52 floating + integer', 'store moments; coefficient solve follows'),
        ('Liu reconstruction', 'scalar, augmented vector/matrix and count', '21 floating + integer', 'Shepard output taken from matrix element'),
        ('MFM gradients', 'two vectors and matrix', '15', 'apply query inverse moment matrix'),
        ('MFM limiter', 'neighbor and reconstructed extrema', '20', 'limiter ratios and clamps'),
        ('MFM coarse stencil', 'one sum per cell offset', '27', 'identity'),
        ('MFM flux', 'mass, momentum, energy', '5', 'identity'),
        ('Signal speed or nearest neighbor', 'best value, ID and validity', 'one floating + integer + bool', 'evaluate selected neighbor; empty/self policy'),
        ('Polynomial moments', 'one coefficient and count per work item', 'one floating + integer', 'store one of P(P+1)/2 coefficients'),
    ]

    report = f'''# Native operator accumulator survey

The existing Warp code supports a reusable native neighborhood traversal with operation-specific query preparation, reference loading, accumulation, combination and finalization. Most state is small. The important exceptions are large moment or polynomial outputs, mixed extrema, dependent traversals and values retained per edge. C++ pointers can represent absent fields and outputs directly.

## Scope and counts

The primary core source is the native experiment worktree at `{data['repositories']['core']['commit']}`. The frontend source is `/home/lu26029/dev/warpSPH` at `{data['repositories']['frontend']['commit']}`. Both source trees were clean when the inventory was first collected. The current core checkout at `{data['core_reference']['commit']}` was also compared read-only.

| Source | Literal wp files | Ordinary wp.func definitions | wp.kernel definitions | Custom wp.func_grad definitions | Total definitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core worktree | {counts['core']['wp_files']} | 406 | 81 | 10 | 497 |
| Frontend | {counts['frontend']['wp_files']} | 178 | 96 | 0 | 274 |
| Total | 95 | 584 | 177 | 10 | 771 |

There are 172 relevant files: 95 `wp_*.py` files plus 77 other files containing decorated Warp definitions. Of the 771 definitions, 390 live in literal `wp_` files and 381 live elsewhere. The latter include CRK moments, acceleration, work, viscosity, reconstruction, Riemann and EOS code, so a filename-only survey would miss substantial implementation work. Ten core files are aliases, type declarations or inactive/scaffolding code with no decorated definitions.

Definitions are counted at their source locations, including overloads and nested kernel factories, rather than once per runtime instantiation. Generated Warp reverse-mode kernels are not additional source definitions. Python wrappers in the relevant files are indexed separately. Indexed assignments in the inventory include local vector/matrix writes and input adjoints; they are not all global output stores.

The [operation catalog]({catalog_target}) contains 113 reviewed operation/pass records: 46 core and 67 frontend. Range, adjacency, serial and tiled launch functions normally share one record. Different schemes, JVP passes, construction utilities and dependent stages can have separate records, so this is not a count of distinct public APIs. Every relevant file has an assessment. The [function inventory]({inventory_target}) records every decorated definition, arguments, indexed loads, updates, self-reassignments, returns, stores and reduction calls, with file hashes and links to local operation records.

## Accumulator sizes

`D` denotes spatial dimension, `F` the number of field components and `P` the polynomial basis size. Counts below describe reduced state, excluding query/reference records and pair temporaries. They do not predict physical register counts. The 3D column describes component sizes; it does not imply every frontend operation supports 3D.

| Operation | Reduced state | Components in 3D | Final processing |
| --- | --- | --- | --- |
'''
    report += '\n'.join('| '+' | '.join(row)+' |' for row in size_summary)
    report += f'''

CRK moments contain `1 + 2D + 2D² + D³` floating-point components: 52 in 3D, plus a neighbor count. Their JVP contains another 52 tangent components in its separate pass. These are full stored tensors, without assuming symmetry compression. The coefficient solve produces `A`, `B`, `gradA` and `gradB`: 16 floating-point components in 3D.

Polynomial fitting is the important unbounded case. Its current moment kernel assigns one work item per query and packed lower-triangle coefficient, accumulating one scalar plus a count. A 3D quadratic basis has P=10 and 55 moment coefficients; higher orders grow further. A universal one-thread-per-query tuple containing the entire polynomial system would reverse an intentional scheduling choice. Preserve coefficient/block scheduling as another execution policy.

The specialized dynGPUSPH shift is currently 2D. It combines 24 floating-point components, two counts and one boolean, using sums, minima, maxima and OR. Its finalization performs weighted ratios, a small matrix inverse, normalization, collision handling, limiting and pressure/velocity updates. This is more involved than the ordinary core operators, but still fits a structured accumulator with an explicit combiner.

## Reduction and output patterns

| Primary pattern | Reviewed records | Required behavior |
| --- | ---: | --- |
| Additive accumulation with direct or simple scaled output | {categories['sum']} | Componentwise sum, then store or apply query-only scaling |
| Additive accumulation followed by final processing | {categories['finalize']} | Ratios, reciprocals, powers or matrix application after reduction |
| Mixed accumulation | {categories['mixed']} | Sum combined with min/max/OR, followed by operation-specific finalization |
| Extrema or boolean neighborhood state | 5 | Per-field min/max/OR and correct identities |
| Winner selection | 4 | Value, winning index, validity and explicit tie policy |
| Dependent traversals or stage work | 4 | Preserve dependencies between completed neighborhood results |
| Edge outputs or edge-producing operator modes | 6 | Optional edge sink and edge-identity/storage contract |
| Search and lookup construction | 8 | Count/allocate/collect or synthetic lattice construction |
| Global or segmented reductions | 2 | Separate global/segment scheduling |
| Additional pointwise solves and fields | 2 | No neighborhood traversal |

These are primary categories, not disjoint semantic capabilities. For example, an acceleration operator can accumulate a query output while optionally writing two vectors per edge, and a closure matvec has a simple scalar sum but consumes retained edge weights.

The {additive} additive records support the proposed abstraction. They do not imply that {additive} operations can be ported mechanically: formulas, supported ranks/schemes, optional corrections and output layouts still need their numerical contracts preserved. Three additional records mix additive quantities with extrema or boolean state.

## Template requirements

A small set of explicit types and methods is sufficient:

| Contract | Responsibility |
| --- | --- |
| Query state and preparation | Load query-only fields, corrections and kernel constants once; handle query masks and nonzero identities |
| Reference requirements and loader | Declare needed fields; supply records from tensor arrays, packed data or shared memory |
| Pair geometry requirements | Compute displacement, distance, kernel value/gradient/Hessian only when required; preserve support and direction rules |
| Accumulator identity | Define zero sums, initialized extrema and invalid winners correctly |
| Interaction callable | Update a directed query accumulator from prepared query/reference records and geometry |
| Combine | Merge partial state with componentwise sum, min, max, OR or winner selection |
| Finalize | Transform the complete reduced state into a result; may return several output fields |
| Store policy | Write query results; support optional outputs independently of accumulator shape |
| Optional edge sink | Store requested pair values when an addressable edge layout exists |
| Traversal and execution | Enumerate CSR/cells/clusters/decoded rows and select CPU, cooperative GPU or component/block schedules |

Keep data loading outside interaction arithmetic so clustered paths can stage one reference record and reuse it for several queries. Geometry and cheap eligibility data can be loaded before expensive payloads where the execution strategy benefits. CRK force calculations need correction data from both particles, while an ordinary corrected scalar gradient primarily uses query correction coefficients. A universal particle record loading every possible field would waste bandwidth and shared memory.

Finalization must run once after all cells, clusters, tiles and lane partials contributing to a query have been combined. A whole-operation dependency on another particle's finalized result requires a later pass. Barecasco detection, for example, computes and normalizes a cover vector before a second neighborhood cone count. The template should support an explicit pass sequence instead of pretending this is a single independent reduction.

Directed gather remains the default: reference data is read-only and each query owns its output. The pair callable describes a directed contribution. It does not require pairwise scatter or atomics into both particles. Floating-point sum reassociation can change low bits, so correctness comparisons should use appropriate tolerances unless a reproducible reduction mode is deliberately provided.

Template the operation/loader/traversal types and the dimensions needed for useful specialization. A small named functor can expose accumulator and requirement types while accepting a templated callable or suitable lambda for interaction arithmetic. Compile-time traits can eliminate unused loads. Keep less important per-launch choices as uniform runtime options to control wheel size, compilation time and the number of kernel variants.

Shared execution and loader contracts belong in the core's existing `native/csrc` hierarchy. Frontend operation functors and formulas can remain beside their modules in `native/csrc_<function>`, following the agreed repository layout. This survey adds no runtime dispatch or simulation checks.

## Optional inputs and outputs

C++, CUDA and HIP accept typed null pointers as kernel arguments and members of pointer views. Python `None` can map to `nullptr` at the wrapper boundary. An absent output pointer can mean that field is not stored. No dummy tensor is necessary to satisfy the language.

For example, a uniform optional scalar field can use:

```cpp
template <class T>
struct OptionalScalarField {{
    const T* values = nullptr;
    T fallback = T(1);

    SPH_HD T load(int64_t index) const {{
        return values ? values[index] : fallback;
    }}
}};
```

Derived defaults such as volume `m/rho` belong in the relevant loader rather than a constant fallback. A correction bundle can be absent as a group; when present, its required members must be supplied. Inputs with several meaningful modes still need a selector where presence alone cannot distinguish the behavior. Optional host tensors can use a host optional representation; device arguments should use simple pointer/scalar views rather than depend on host standard-library optional types.

The contract must guard an optional pointer before dereferencing it. A common hot mode can be selected once at dispatch and compiled with `if constexpr`, removing its optional loads and branches. A uniform runtime pointer test is also legal. Omitting an output store alone does not guarantee the corresponding arithmetic disappears: requested-output traits are needed when that is a performance goal.

The present native wrapper does not yet expose a general nullable-field contract: `RowArgs` has raw pointers and `IndexView` already defaults its data pointer to null, while the Torch bridge currently marshals a vector of stable Tensor handles. Wrapper/schema changes are needed to represent absence explicitly and retain present tensors for the asynchronous operation's lifetime. A non-null GPU pointer must refer to device-accessible storage. These are ordinary pointer/API rules, not a need for content scans or CPU copies in each operation.

## Porting hazards

1. **Incompressible alpha has a cell-level nonlinear step.** {ref('frontend','src/warpSPH/modules/incompressible/wp_alpha.py','computeAlpha_Func_Adjacency_first','The current adjacency function')} computes `area_i/m_i * dot(sumA,sumA) + area_i*sumB` inside its cell loop, then adds those results. CSR supplies one range; a grid supplies several. Squaring the complete vector sum includes cross-cell terms absent from a sum of squared partial vectors. Finalizing once over the full neighborhood therefore changes the current grid result. This is a source-level semantic discrepancy, not a newly executed failing test; decide and verify the intended behavior explicitly during porting.

2. **Roots already have a correct raw/finalized split.** {ref('frontend','src/warpSPH/modules/adaptiveSupport/wp_psi0.py','computePsi0_Func_Adjacency_raw','Psi0 raw accumulation')} sums both quantities across cells and lanes before dimension roots. Material psi follows the same pattern. Preserve this boundary. {ref('frontend','src/warpSPH/modules/artificialCompressible/wp_jstSwitch.py','computeJstSwitch_Kernel','JST')} stores two raw sums and its Python wrapper performs a clamped ratio; that processing can move into native finalization.

3. **Selection carries policy as well as a value.** Signal-speed, characteristic-speed and nearest-normal routines carry `found`, a value and an index, then reevaluate the selected interaction outside the loop for Warp AD. Strict comparisons retain the first winner in the current traversal. Parallel combination needs a specified tie rule and empty/self behavior. Preserving traversal-first ties requires an ordinal; changing to stable-particle-ID ties is a deliberate semantic choice. Future backward passes must follow the selected winner and finalization derivative.

4. **Retained edges are a data-lifetime issue.** Compressible/CRK acceleration can write pressure and viscosity vectors per edge; balancing writes a fraction; implicit shifting writes kernel/gradient/Hessian terms; MFM closure stores and rereads edge weights. A list-free path needs recomputation, different compact storage or a separately supported retained mode. Passing a null edge sink is useful only when the downstream computation also supports its absence.

5. **Some recomputation paths already exist.** {ref('frontend','src/warpSPH/modules/crk/accel.py','computeCrkSPHPair','The shared CRK pair formula')} is reused by force-only acceleration and {ref('frontend','src/warpSPH/modules/crk/work.py','_crkWorkKernel','the stage work kernel')}; compressible SPH has the same pattern. Work can reduce directly into a query rate or store/add a vector per edge. The direct path already demonstrates how a second force traversal can avoid retained pair accelerations. These are Warp implementations despite some source docstrings calling them native. They remain CSR-based and require porting plus stage/force-balance verification for clustered execution.

6. **The current core checkout and native worktree differ in tensor layout.** All decorated core definitions match after accounting for seven moved primal files except the helpers in `math/wp_divdot.py` and `math/wp_outerTensorProduct.py`. The worktree flattens the tensor outer product as field component then derivative axis; current core uses derivative axis then field component. The two checkouts also reverse which contraction layout `dotMode` selects. Accumulator sizes are unchanged, but the native interface must name its layout and wrappers must map it deliberately. This survey does not alter either checkout.

7. **AD is separate from primal traversal reuse.** Manual local adjoints and explicit JVP/HVP code appear in the inventory, but generated Warp adjoints are not manually implemented native backwards. Winner selection, nonlinear finalization and transpose/scatter ownership require their own derivative rules. A reusable primal traversal is useful without implying an automatic complete AD chain.

## Complexity and implementation sequence

The reviewed records have 43 low, 48 medium and 22 high source-based complexity ratings. These are qualitative integration ratings, not elapsed-time estimates. Many high-rated records have tiny accumulators; their cost lies in Riemann/reconstruction physics or edge/stage dependencies. Math helpers and launch duplication should not be treated as 771 separate operators to port.

The traversal framework itself is moderate work and can remain a few focused headers. The evidence argues against a single generic sum tuple or a runtime list of field descriptors. Named accumulator and loader types with an explicit combiner cover the observed cases more clearly. Large polynomial outputs need a different execution schedule rather than a more elaborate pair interface.

Prove the contract with density/gradient, a two-sum ratio or root, CRK moments, an arg selector and mixed MFM extrema. Then cover an edge sink and a dependent two-pass operation. Standalone correctness checks should compare CSR and alternate traversals, tile partitions, empty/masked queries, optionals, selection ties and output layouts. Such checks belong in verification/tests; they need not add knobs or scans to simulation calls.

The agreed backend implementation order remains sorting, native CSR32/64, clustered/cell execution, then two-bit decoding. Shared CSR permits incremental coexistence with Warp. Clustered simulation coverage exercises the complete native operator stack and its stage dependencies before compression becomes another traversal choice.

## Source catalog and reproducibility

The [catalog]({catalog_target}) gives the accumulated values, combination, final processing, stored outputs and loading requirements for every operation record, then an assessment of every relevant file. The JSON includes every decorated source definition and exact local expressions, plus the current-core comparison. The extractor does not import either package or run a kernel. It validates file assessment coverage and operation anchor names, and records source hashes so concurrent edits can be detected.

Recreate the inventory with the script and assessments in `experiments/operator_survey`. Rerender the report with `render_report.py`. The numerical kernels, frontend source and active simulation paths are unchanged by this survey.
'''
    (directory / (stem+'.md')).write_text(report)

    catalog = ['# Native operator accumulator catalog', '',
        'This catalog accompanies the accumulator survey. D is dimension, F is field component count, and P is polynomial basis size. Counts describe reduced state rather than total registers. Each record groups related range, adjacency and launch definitions where applicable. Source links point to the reviewed checkouts.', '']
    for package in ['core','frontend']:
        catalog += ['## '+('Core operations' if package=='core' else 'Frontend operations'),'']
        for operation in data['operations']:
            if operation['package'] != package:
                continue
            catalog += ['### '+operation['name'].replace('-',' '), '',
                source_link(package, operation['path'], anchor(package,operation['path'],operation['anchors'][0])['line']), '',
                '- Accumulates: '+operation['accumulators']+'.',
                '- Components: '+operation['scalar_components']+'.',
                '- Combines: '+operation['combine']+'.',
                '- Finalizes: '+operation['finalize']+'.',
                '- Stores: '+operation['stores']+'.',
                '- Loads: '+operation['reference_loads']+'.',
                '- Primary pattern and complexity: '+operation['category']+'; '+operation['complexity']+'.']
            if operation['notes']:
                catalog += ['- Detail: '+operation['notes']]
            catalog += ['']
    catalog += ['## File coverage', '',
        'All literal wp files and every other file with a decorated definition are listed below. The function inventory contains individual definition locations and exact return/update/store expressions. A file with zero decorated definitions contributes no additional Warp operation.', '']
    for package in ['core','frontend']:
        catalog += ['### '+package.capitalize(), '',
            '| File | Definitions | Family | Accumulation and outputs |',
            '| --- | ---: | --- | --- |']
        for item in data['files']:
            if item['package'] != package:
                continue
            assessment = item['assessment']
            detail = ('Operation records: '+', '.join(assessment['operations'])+'.') if assessment['operations'] else (
                assessment['accumulator_summary']+' '+assessment['processing']+' Stores: '+assessment['stores']+'.')
            label = item['path'].split('/',2)[2]
            values = [source_link(package,item['path'],label=label),str(item['decorated_definitions']),assessment['family'],detail]
            catalog.append('| '+' | '.join(value.replace('|','\\|').replace('\n',' ') for value in values)+' |')
        catalog += ['']
    catalog += ['## Definition coverage', '',
        'Each decorated source definition is linked below. Operation links follow local call relationships to reviewed anchors; a local helper may have no operation link and is described by its file assessment. The JSON retains the exact evidence and distinguishes source roles. Overloads with the same name remain separate by line.', '',
        '| Source definition | Decorator | Role | Linked passes |',
        '| --- | --- | --- | --- |']
    for item in data['functions']:
        label=item['package']+':'+item['path'].split('/',2)[2]+':'+str(item['line'])+' '+item['name']
        values=[source_link(item['package'],item['path'],item['line'],label),item['kind'],item['source_role'],', '.join(item['operation_links']) or 'local helper or file assessment']
        catalog.append('| '+' | '.join(value.replace('|','\\|').replace('\n',' ') for value in values)+' |')
    (directory/catalog_name).write_text('\n'.join(catalog)+'\n')
    print(json.dumps({'report':str(directory/(stem+'.md')),'catalog':str(directory/catalog_name),
                      'records':len(data['operations']),'files':len(data['files']),
                      'definitions':len(data['functions'])}))


if __name__ == '__main__':
    main()
