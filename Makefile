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
INCLUDES  += -I./include -I./external/cccl/libcudacxx/include

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
	$(BUILDDIR)/test_mathops \
	$(BUILDDIR)/test_ops \
	$(BUILDDIR)/test_fold \
	$(BUILDDIR)/test_axred \
	$(BUILDDIR)/test_valueform \
	$(BUILDDIR)/test_compare \
	$(BUILDDIR)/test_ellipsis \
	$(BUILDDIR)/test_iterate \
	$(BUILDDIR)/test_subperm \
	$(BUILDDIR)/test_takealong \
	$(BUILDDIR)/test_static_extent \
	$(BUILDDIR)/test_slice \
	$(BUILDDIR)/test_broadcast \
	$(BUILDDIR)/test_api \
	$(BUILDDIR)/test_half \
	$(BUILDDIR)/test_promote \
	$(BUILDDIR)/test_strides \
	$(BUILDDIR)/test_dynamic \
	$(BUILDDIR)/test_distance_l1 \
	$(BUILDDIR)/test_pull \
	$(BUILDDIR)/test_posdef \
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

clean:
	$(DEL) $(TESTS) $(EXAMPLES)

.PHONY: all test run-test examples run-examples clean

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
	@ $(BUILDDIR)/ex_$* >/dev/null && echo "  PASS  ex_$*" || echo "  FAIL  ex_$*"

# test_cuda needs the fake CUDA runtime on the include path.
$(BUILDDIR)/test_cuda: tests/test_cuda.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Itests/fakecuda $(TESTFLAGS) -o $@ $<

run-%: $(BUILDDIR)/test_%
	@ $(BUILDDIR)/test_$* && echo "  PASS  test_$*" || echo "  FAIL  test_$*"

########################################################################
# 	Messages
########################################################################

verb.build:      ; $(call verb, "Building tests...")
verb.build.done: ; $(call verb, "Building tests: Done.")
verb.run:        ; $(call verb, "Running tests...")
verb.run.done:   ; $(call verb, "Running tests: Done.")
