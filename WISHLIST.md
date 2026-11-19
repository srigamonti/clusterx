# CELL wishlist

This file records possible improvements and open design questions for CELL.
Entries are proposals, not active tasks or commitments to a particular design.
Their order does not establish priority. Working guidance for agents lives in
[AGENTS.md](AGENTS.md).

For new entries, describe the motivation, possible approaches and alternatives,
expected benefits, compatibility concerns, and status. Useful statuses are
**Idea**, **Needs investigation**, and **Agreed direction**. An agreed direction
still needs a concrete task scope before implementation.

## Reduce coupling between the domain model and ASE

**Status:** Needs investigation.

**Motivation:** The maintainer considers the heavy reliance on ASE's `Atoms`
object a design limitation. The declared inheritance chain is
`Atoms -> ParentLattice -> SuperCell -> Structure`, while the implementation also
uses internal ASE objects and delegation. Geometry, allowed species, site types,
and decorations have distinct responsibilities whose ownership and mutation
semantics deserve examination.

**Possible approaches:** Investigate a gradual move toward composition, with
explicit CELL domain objects and ASE conversion or adapters at integration
boundaries. Compare this with retaining inheritance while first clarifying data
ownership and supported operations. Determine the practical benefits before
choosing a replacement architecture.

**Expected benefits:** Clearer responsibilities and invariants, easier testing,
and fewer opportunities for inconsistent state across representations and caches.

**Compatibility concerns:** Existing Python workflows may rely on inherited ASE
methods, object identity, copying, and mutation behavior. A migration must account
for public APIs, saved objects, ASE I/O and calculators, CLI workflows, and the
scientific behavior of downstream algorithms.

**Open questions:**

- Which inherited ASE operations are part of the supported CELL API?
- Which objects should own geometry, lattice metadata, and decorations?
- What incremental steps would deliver useful improvements without a broad
  rewrite, and how would compatibility be maintained?

## Reconsider the organization of cluster construction code

**Status:** Idea.

**Motivation:** Cluster-pool construction is central to CELL. The maintainer has
suggested that `clusterx/clusters/`, currently a subpackage, might be better
organized as a module.

**Possible approaches:** Assess consolidating the cluster, orbit, and pool code
into a module. Compare this with keeping the subpackage and improving its internal
boundaries and public exports. Choose based on clarity, size, dependencies, and
likely future development.

**Expected benefits:** Easier navigation and a clearer public interface for
cluster construction and manipulation.

**Compatibility concerns:** Preserve or provide a migration for imports such as
`clusterx.clusters.cluster` and `clusterx.clusters.clusters_pool`, including class
paths embedded in existing pickles. Verify that cluster ordering, symmetry
orbits, multiplicities, and correlation/model behavior remain consistent.

**Open questions:**

- Would a single module improve readability as the functionality grows?
- What should the public interface expose for `Cluster`, `ClustersPool`, and
  `ClusterOrbit`?
- What migration strategy would support existing scripts and serialized objects?

## Supercell creation from a parent lattice

The SuperCell object currently uses ASEs make_supercell. Since various things
rely on the specific ordering of crystal sites of the created supercell, 
any changes to the underlying ASE function or its usage could affect downstream 
algorithms and data consistency. This is bad because it can introduce subtle, hard-to-detect
bugs that compromise the reliability of downstream computations and analyses. 
As an example, DerivativeStructures objects store the parent lattice,
the (integer) transformation matrix, and the tuple of integers  indicating what species occupy each site in the supercell. The actual
positions are not explicitly stored, and the correct structure relies on robust 
recreation of the supercell from the parent lattice and transformation matrix.
So we need to ensure a consistent and predictable site ordering when creating supercells by implementing a custom, CELL native supercell construction routine.

**Possible approaches:** Investigate wrapping or adapting ASE's `make_supercell` to enforce a consistent site ordering, or implement a custom supercell construction routine that guarantees the desired ordering.

**Expected benefits:** Improved robustness and predictability of supercell creation, reducing the risk of subtle bugs in downstream algorithms that depend on site ordering.

**Compatibility concerns:** Ensure that any changes preserve the current behavior for existing scripts and serialized objects. Verify that the resulting supercells maintain the expected site ordering and that dependent algorithms continue to function correctly.

