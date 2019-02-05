from collections import OrderedDict
from itertools import combinations, product

import cgen
import numpy as np

from devito.cgen_utils import ccode
from devito.dle import BlockDimension, fold_blockable_tree, unfold_blocked_tree
from devito.dle.backends import (BasicRewriter, Ompizer, dle_pass, simdinfo,
                                 get_simd_flag, get_simd_items)
from devito.exceptions import DLEException
from devito.ir.iet import (Call, Expression, Iteration, List, HaloSpot, PARALLEL,
                           REMAINDER, FindSymbols, FindNodes, FindAdjacent,
                           IsPerfectIteration, MapNodes, Transformer, compose_nodes,
                           retrieve_iteration_tree, make_efunc)
from devito.logger import perf_adv
from devito.tools import as_tuple, flatten


class AdvancedRewriter(BasicRewriter):

    _parallelizer = Ompizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halospots(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._minimize_remainders(state)

    @dle_pass
    def _loop_wrapping(self, iet, state):
        """
        Emit a performance warning if WRAPPABLE Iterations are found,
        as these are a symptom that unnecessary memory is being allocated.
        """
        for i in FindNodes(Iteration).visit(iet):
            if not i.is_Wrappable:
                continue
            perf_adv("Functions using modulo iteration along Dimension `%s` "
                     "may safely allocate a one slot smaller buffer" % i.dim)
        return iet, {}

    @dle_pass
    def _optimize_halospots(self, iet, state):
        """
        Optimize the HaloSpots in ``iet``.

        * Remove all USELESS HaloSpots;
        * Merge all HOISTABLE HaloSpots with their root HaloSpot, thus
          removing redundant communications and anticipating communications
          that will be required by later Iterations.
        """
        # Drop USELESS HaloSpots
        mapper = {hs: hs.body for hs in FindNodes(HaloSpot).visit(iet) if hs.is_Useless}
        iet = Transformer(mapper, nested=True).visit(iet)

        # Handle `hoistable` HaloSpots
        mapper = {}
        for halo_spots in MapNodes(Iteration, HaloSpot).visit(iet).values():
            root = halo_spots[0]
            halo_schemes = [hs.halo_scheme.project(hs.hoistable) for hs in halo_spots[1:]]
            mapper[root] = root._rebuild(halo_scheme=root.halo_scheme.union(halo_schemes))
            mapper.update({hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(hs.hoistable))
                           for hs in halo_spots[1:]})
        iet = Transformer(mapper, nested=True).visit(iet)

        # At this point, some HaloSpots may be empty (i.e., no communications required)
        # Such HaloSpots can be dropped and their body might be squashable with that
        # of an adjacent HaloSpot
        mapper = {}
        for v in FindAdjacent(HaloSpot).visit(iet).values():
            for g in v:
                root = g[0]
                for i in g[1:]:
                    if i.is_empty:
                        rebuilt = mapper.get(root, root)
                        body = List(body=as_tuple(root.body) + (i.body,))
                        mapper[root] = rebuilt._rebuild(body=body)
                        mapper[i] = None
                    else:
                        root = i
        # Finally, any leftover empty HaloSpot should be dropped
        mapper.update({i: i.body for i in FindNodes(HaloSpot).visit(iet)
                       if i.is_empty and i not in mapper})
        iet = Transformer(mapper, nested=True).visit(iet)

        return iet, {}

    @dle_pass
    def _loop_blocking(self, iet, state):
        """Apply loop blocking to PARALLEL Iteration trees."""
        exclude_innermost = not self.params.get('blockinner', False)
        ignore_heuristic = self.params.get('blockalways', False)
        noinline = self._compiler_decoration('noinline', cgen.Comment('noinline?'))

        # Make sure loop blocking will span as many Iterations as possible
        iet = fold_blockable_tree(iet, exclude_innermost)

        mapper = {}
        efuncs = []
        block_dims = []
        for tree in retrieve_iteration_tree(iet):
            # Is the Iteration tree blockable ?
            iterations = [i for i in tree if i.is_Parallel]
            if exclude_innermost:
                iterations = [i for i in iterations if not i.is_Vectorizable]
            if len(iterations) <= 1:
                continue
            root = iterations[0]
            if not IsPerfectIteration().visit(root):
                # Illegal/unsupported
                continue
            if not tree.root.is_Sequential and not ignore_heuristic:
                # Heuristic: avoid polluting the generated code with blocked
                # nests (thus increasing JIT compilation time and affecting
                # readability) if the blockable tree isn't embedded in a
                # sequential loop (e.g., a timestepping loop)
                continue

            # Apply loop blocking to `tree`
            interb = []
            intrab = []
            for i in iterations:
                d = BlockDimension(i.dim, name="%s%d_block" % (i.dim.name, len(mapper)))
                # Build Iteration over blocks
                interb.append(Iteration([], d, d.symbolic_max, offsets=i.offsets,
                                        properties=PARALLEL))
                # Build Iteration within a block
                intrab.append(i._rebuild([], limits=(d, d+d.step-1, 1), offsets=(0, 0)))
                # Record that a new BlockDimension has been introduced
                block_dims.append(d)

            # Construct the blocked tree
            blocked = compose_nodes(interb + intrab + [iterations[-1].nodes])
            blocked = unfold_blocked_tree(blocked)

            # Promote to a separate Callable
            dynamic_parameters = flatten((bi.dim, bi.dim.symbolic_size) for bi in interb)
            efunc = make_efunc("bf%d" % len(mapper), blocked, dynamic_parameters)
            efuncs.append(efunc)

            # Compute the iteration ranges
            ranges = []
            for i, bi in zip(iterations, interb):
                maxb = i.symbolic_max - (i.symbolic_size % bi.dim.step)
                ranges.append(((i.symbolic_min, maxb, bi.dim.step),
                               (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

            # Build Calls to the `efunc`
            body = []
            for p in product(*ranges):
                dynamic_parameters_mapper = {}
                for bi, (m, M, b) in zip(interb, p):
                    dynamic_parameters_mapper[bi.dim] = (m, M)
                    dynamic_parameters_mapper[bi.dim.step] = (b,)
                body.append(efunc.make_call(dynamic_parameters_mapper))

            # Build indirect Call to the `efunc` Calls
            dynamic_parameters = [i.dim for i in iterations]
            dynamic_parameters.extend([bi.dim.step for bi in interb])
            efunc = make_efunc("f%d" % len(mapper), body, dynamic_parameters)
            efuncs.append(efunc)

            # Track everything to ultimately transform the input `iet`
            mapper[root] = efunc.make_call()

        iet = Transformer(mapper).visit(iet)

        return iet, {'dimensions': block_dims, 'efuncs': efuncs}

    @dle_pass
    def _simdize(self, nodes, state):
        """
        Add compiler-specific or, if not available, OpenMP pragmas to the
        Iteration/Expression tree to emit SIMD-friendly code.
        """
        ignore_deps = as_tuple(self._compiler_decoration('ignore-deps'))

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            for i in vector_iterations:
                handle = FindSymbols('symbolics').visit(i)
                try:
                    aligned = [j for j in handle if j.is_Tensor and
                               j.shape[-1] % get_simd_items(j.dtype) == 0]
                except KeyError:
                    aligned = []
                if aligned:
                    simd = Ompizer.lang['simd-for-aligned']
                    simd = as_tuple(simd(','.join([j.name for j in aligned]),
                                    simdinfo[get_simd_flag()]))
                else:
                    simd = as_tuple(Ompizer.lang['simd-for'])
                mapper[i] = i._rebuild(pragmas=i.pragmas + ignore_deps + simd)

        processed = Transformer(mapper).visit(nodes)

        return processed, {}

    @dle_pass
    def _parallelize(self, iet, state):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit parallel code
        """
        def key(i):
            return i.is_ParallelRelaxed and not (i.is_Elementizable or i.is_Vectorizable)
        return self._parallelizer(key).make_parallel(iet)

    @dle_pass
    def _minimize_remainders(self, nodes, state):
        """
        Reshape temporary tensors and adjust loop trip counts to prevent as many
        compiler-generated remainder loops as possible.
        """
        # The innermost dimension is the one that might get padded
        p_dim = -1

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            if not vector_iterations or len(vector_iterations) > 1:
                continue
            root = vector_iterations[0]
            if root.tag is None:
                continue

            # Padding
            writes = [i.write for i in FindNodes(Expression).visit(root)
                      if i.write.is_Array]
            padding = []
            for i in writes:
                try:
                    simd_items = get_simd_items(i.dtype)
                except KeyError:
                    # Fallback to 16 (maximum expectable padding, for AVX512 registers)
                    simd_items = simdinfo['avx512f'] / np.dtype(i.dtype).itemsize
                padding.append(simd_items - i.shape[-1] % simd_items)
            if len(set(padding)) == 1:
                padding = padding[0]
                for i in writes:
                    padded = (i._padding[p_dim][0], i._padding[p_dim][1] + padding)
                    i.update(padding=i._padding[:p_dim] + (padded,))
            else:
                # Padding must be uniform -- not the case, so giving up
                continue

            # Dynamic trip count adjustment
            endpoint = root.symbolic_max
            if not endpoint.is_Symbol:
                continue
            condition = []
            externals = set(i.symbolic_shape[-1] for i in FindSymbols().visit(root)
                            if i.is_Tensor)
            for i in root.uindices:
                for j in externals:
                    condition.append(root.symbolic_max + padding < j)
            condition = ' && '.join(ccode(i) for i in condition)
            endpoint_padded = endpoint.func('_%s' % endpoint.name)
            init = cgen.Initializer(
                cgen.Value("const int", endpoint_padded),
                cgen.Line('(%s) ? %s : %s' % (condition,
                                              ccode(endpoint + padding),
                                              endpoint))
            )

            # Update the Iteration bound
            limits = list(root.limits)
            limits[1] = endpoint_padded.func(endpoint_padded.name)
            rebuilt = list(tree)
            rebuilt[rebuilt.index(root)] = root._rebuild(limits=limits)

            mapper[tree[0]] = List(header=init, body=compose_nodes(rebuilt))

        processed = Transformer(mapper).visit(nodes)

        return processed, {}


class AdvancedRewriterSafeMath(AdvancedRewriter):

    """
    This Rewriter is slightly less aggressive than AdvancedRewriter, as it
    doesn't drop denormal numbers, which may sometimes harm the numerical precision.
    """

    def _pipeline(self, state):
        self._optimize_halospots(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._minimize_remainders(state)


class SpeculativeRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halospots(state)
        self._loop_wrapping(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._minimize_remainders(state)

    @dle_pass
    def _nontemporal_stores(self, nodes, state):
        """
        Add compiler-specific pragmas and instructions to generate nontemporal
        stores (ie, non-cached stores).
        """
        pragma = self._compiler_decoration('ntstores')
        fence = self._compiler_decoration('storefence')
        if not pragma or not fence:
            return {}

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            for i in tree:
                if i.is_Parallel:
                    mapper[i] = List(body=i, footer=fence)
                    break
        processed = Transformer(mapper).visit(nodes)

        mapper = {}
        for tree in retrieve_iteration_tree(processed):
            for i in tree:
                if i.is_Vectorizable:
                    mapper[i] = List(header=pragma, body=i)
        processed = Transformer(mapper).visit(processed)

        return processed, {}


class CustomRewriter(SpeculativeRewriter):

    passes_mapper = {
        'denormals': SpeculativeRewriter._avoid_denormals,
        'optcomms': SpeculativeRewriter._optimize_halospots,
        'wrapping': SpeculativeRewriter._loop_wrapping,
        'blocking': SpeculativeRewriter._loop_blocking,
        'openmp': SpeculativeRewriter._parallelize,
        'simd': SpeculativeRewriter._simdize,
    }

    def __init__(self, nodes, passes, params):
        try:
            passes = passes.split(',')
            if 'openmp' not in passes and params['openmp']:
                passes.append('openmp')
        except AttributeError:
            # Already in tuple format
            if not all(i in CustomRewriter.passes_mapper for i in passes):
                raise DLEException
        self.passes = passes
        super(CustomRewriter, self).__init__(nodes, params)

    def _pipeline(self, state):
        for i in self.passes:
            CustomRewriter.passes_mapper[i](self, state)
