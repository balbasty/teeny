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
	$(BUILDDIR)/test_iterate \
	$(BUILDDIR)/test_subperm \
	$(BUILDDIR)/test_static_extent \
	$(BUILDDIR)/test_dynamic \
	$(BUILDDIR)/test_distance_l1 \
	$(BUILDDIR)/test_pull \
	$(BUILDDIR)/test_posdef \
	$(BUILDDIR)/test_cuda

########################################################################
# 	Public Targets
########################################################################

all: test

test: verb.build $(TESTS) verb.build.done

run-test: verb.run $(TESTS:$(BUILDDIR)/test_%=run-%) verb.run.done

clean:
	$(DEL) $(TESTS)

.PHONY: all test run-test clean

########################################################################
# 	Rules
########################################################################

$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)

$(BUILDDIR)/test_%: tests/test_%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTFLAGS) -o $@ $<

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
