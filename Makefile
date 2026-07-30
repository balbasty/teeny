define verb
	@ echo "_____________________________________________________________"
	@ echo ""
	@ echo "        " $(1)
	@ echo "_____________________________________________________________"
	@ echo ""
endef

DEL       ?= rm -f
MKDIR     ?= mkdir -p
BUILDDIR  ?= ./build
CXXFLAGS  += -std=c++17
# CCCL (libcudacxx: cuda::std::mdspan) — selected automatically, nothing to set:
#   * host compilers and nvcc <= 12.x  -> the vendored CCCL (v2.8.2), which covers
#     the whole CUDA 11.1-12.9 range;
#   * nvcc >= 13                        -> the toolkit's own bundled CCCL (a 3.x that
#     works with teeny), so the vendored include is skipped.
# Override CCCL_INC to force a specific CCCL for any compiler. See docs/cuda-compat.md.
_CUDA_MAJOR := $(shell $(CXX) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -1)
ifeq ($(shell [ -n "$(_CUDA_MAJOR)" ] && [ "$(_CUDA_MAJOR)" -ge 13 ] 2>/dev/null && echo 1),1)
CCCL_INC  ?=
else
CCCL_INC  ?= ./external/cccl/libcudacxx/include
endif
INCLUDES  += -I./include $(if $(strip $(CCCL_INC)),-I$(CCCL_INC))

# Stop at the first error (the flag differs between clang and gcc).
CXX_IS_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -qi clang && echo 1)
ifeq ($(CXX_IS_CLANG),1)
TESTFLAGS += -ferror-limit=1
else
TESTFLAGS += -fmax-errors=1
endif
TESTFLAGS += -ftemplate-backtrace-limit=0

# All tests. `test_cuda` additionally needs the malloc-backed fake CUDA runtime.
TESTS = \
	$(BUILDDIR)/test_tensor \
	$(BUILDDIR)/test_math \
	$(BUILDDIR)/test_atomic_alias \
	$(BUILDDIR)/test_atomic_fetch_add \
	$(BUILDDIR)/test_mathops \
	$(BUILDDIR)/test_vecalg \
	$(BUILDDIR)/test_into \
	$(BUILDDIR)/test_reduce_methods \
	$(BUILDDIR)/test_dtype \
	$(BUILDDIR)/test_wrap_layout_tag \
	$(BUILDDIR)/test_wrap_compose \
	$(BUILDDIR)/test_wrap_mdspan \
	$(BUILDDIR)/test_wrap_layout_dup \
	$(BUILDDIR)/test_ops \
	$(BUILDDIR)/test_fold \
	$(BUILDDIR)/test_axred \
	$(BUILDDIR)/test_static_unroll \
	$(BUILDDIR)/test_valueform \
	$(BUILDDIR)/test_compare \
	$(BUILDDIR)/test_ellipsis \
	$(BUILDDIR)/test_to \
	$(BUILDDIR)/test_empty \
	$(BUILDDIR)/test_factory_dtype \
	$(BUILDDIR)/test_reduce_dtype \
	$(BUILDDIR)/test_reduce_compose \
	$(BUILDDIR)/test_iterate \
	$(BUILDDIR)/test_peel_enumerate \
	$(BUILDDIR)/test_peel_recast \
	$(BUILDDIR)/test_peel_shape \
	$(BUILDDIR)/test_anyrank_tail \
	$(BUILDDIR)/test_recast_guard \
	$(BUILDDIR)/test_overlap_guard \
	$(BUILDDIR)/test_reindex \
	$(BUILDDIR)/test_dispatch_index \
	$(BUILDDIR)/test_dispatch_layout \
	$(BUILDDIR)/test_reshape_view \
	$(BUILDDIR)/test_reshape \
	$(BUILDDIR)/test_subperm \
	$(BUILDDIR)/test_takealong \
	$(BUILDDIR)/test_subsample \
	$(BUILDDIR)/test_unfold \
	$(BUILDDIR)/test_scan \
	$(BUILDDIR)/test_index_select \
	$(BUILDDIR)/test_peel_zip \
	$(BUILDDIR)/test_alias \
	$(BUILDDIR)/test_axis_sort \
	$(BUILDDIR)/test_kwargs \
	$(BUILDDIR)/test_kwargs_readers \
	$(BUILDDIR)/test_static_extent \
	$(BUILDDIR)/test_wrap \
	$(BUILDDIR)/test_slice \
	$(BUILDDIR)/test_newaxis \
	$(BUILDDIR)/test_unchecked \
	$(BUILDDIR)/test_assign \
	$(BUILDDIR)/test_subscript \
	$(BUILDDIR)/test_broadcast \
	$(BUILDDIR)/test_broadcast_index \
	$(BUILDDIR)/test_inplace_vectorize \
	$(BUILDDIR)/test_unroll \
	$(BUILDDIR)/test_restrict_fastpath \
	$(BUILDDIR)/test_api \
	$(BUILDDIR)/test_half \
	$(BUILDDIR)/test_promote \
	$(BUILDDIR)/test_strides \
	$(BUILDDIR)/test_dynamic \
	$(BUILDDIR)/test_distance_l1 \
	$(BUILDDIR)/test_pull \
	$(BUILDDIR)/test_posdef \
	$(BUILDDIR)/test_dlpack \
	$(BUILDDIR)/test_cuda

EXAMPLES = \
	$(BUILDDIR)/ex_pull_nd \
	$(BUILDDIR)/ex_distance_transform \
	$(BUILDDIR)/ex_cholesky_solve \
	$(BUILDDIR)/ex_broadcast_affine \
	$(BUILDDIR)/ex_pushpull_adjoint \
	$(BUILDDIR)/ex_batched_inverse

########################################################################
# 	Public Targets
########################################################################

all: test

test: verb.build $(TESTS) verb.build.done

run-test: verb.run $(TESTS:$(BUILDDIR)/test_%=run-%) verb.run.done

examples: verb.build $(EXAMPLES) verb.build.done

run-examples: verb.run $(EXAMPLES:$(BUILDDIR)/ex_%=runex-%) verb.run.done

# Release build of the tests: optimized, assertions stripped via the ISO -DNDEBUG
# (so _TNY_CHECK precondition checks compile out) — the "trust the inputs" build.
# Recursive so the flag change forces a clean rebuild (a bare `.cpp` is unchanged,
# so make would otherwise reuse the default-build binaries).
release:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -O2 -DNDEBUG' run-test

# Hardened build: turn ON the opt-in element-access bounds checks (mdspan, and thus
# teeny, is unchecked by default). Independent of NDEBUG, so it composes with a
# release build; verifies the checks don't reject valid access.
hardened:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -O2 -DTNY_HARDENED' run-test

# Build/run the suite at -std=c++23 (the later -std wins), which exercises the
# C++23-only `operator[]` multidimensional subscript that the default c++17 CI
# cannot reach. teeny targets c++17 but must stay forward-compatible.
cxx23:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -std=c++23' run-test

clean:
	$(DEL) $(TESTS) $(EXAMPLES)

.PHONY: all test run-test examples run-examples release hardened cxx23 clean

########################################################################
# 	Rules
########################################################################

$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)

$(BUILDDIR)/test_%: tests/test_%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTFLAGS) -o $@ $<

$(BUILDDIR)/ex_%: examples/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTFLAGS) -o $@ $<

# batched_inverse uses std::thread
$(BUILDDIR)/ex_batched_inverse: examples/batched_inverse.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -pthread $(TESTFLAGS) -o $@ $<

runex-%: $(BUILDDIR)/ex_%
	@ $(BUILDDIR)/ex_$* >/dev/null && echo "  PASS  ex_$*" || { echo "  FAIL  ex_$*"; exit 1; }

# test_cuda needs the fake CUDA runtime on the include path.
$(BUILDDIR)/test_cuda: tests/test_cuda.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Itests/fakecuda $(TESTFLAGS) -o $@ $<

# test_atomic_fetch_add uses std::thread (#257 host atomic stress test).
$(BUILDDIR)/test_atomic_fetch_add: tests/test_atomic_fetch_add.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -pthread $(TESTFLAGS) -o $@ $<

run-%: $(BUILDDIR)/test_%
	@ $(BUILDDIR)/test_$* && echo "  PASS  test_$*" || { echo "  FAIL  test_$*"; exit 1; }

########################################################################
# 	Messages
########################################################################

verb.build:      ; $(call verb, "Building tests...")
verb.build.done: ; $(call verb, "Building tests: Done.")
verb.run:        ; $(call verb, "Running tests...")
verb.run.done:   ; $(call verb, "Running tests: Done.")
