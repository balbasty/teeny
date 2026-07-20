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

########################################################################
# 	Public Targets
########################################################################

all: test

clean: clean-test

test: verb.build.test \
	test-statix \
	test-xarray \
	test-tensor \
	test-md \
	verb.build.test.done

run-test: verb.run.test \
	run-test-statix \
	run-test-xarray \
	run-test-tensor \
	run-test-md \
	verb.run.test.done

clean-test: verb.clean.test \
	clean-test-statix \
	clean-test-xarray \
	clean-test-tensor \
	clean-test-md \
	verb.clean.test.done

.PHONY: all clean test run-test clean-test
.PHONY: test-statix run-test-statix clean-test-statix
.PHONY: test-xarray run-test-xarray clean-test-xarray
.PHONY: test-tensor run-test-tensor clean-test-tensor
.PHONY: test-md run-test-md clean-test-md

########################################################################
# 	Test Targets
########################################################################

$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)

$(BUILDDIR)/test_%: tests/test_%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTFLAGS) \
	-o $@ $<

TESTS_STATIX = \
	$(BUILDDIR)/test_statix_carray \
	$(BUILDDIR)/test_statix_math \
	$(BUILDDIR)/test_statix_pack \
	$(BUILDDIR)/test_statix_tuple

test-statix: verb.build.test.statix \
	$(TESTS_STATIX) \
	verb.build.test.statix.done

run-test-statix: verb.run.test.statix \
	run-test-statix-carray run-test-statix-math run-test-statix-pack run-test-statix-tuple \
	verb.run.test.statix.done

clean-test-statix:
	$(DEL) $(TESTS_STATIX)

test-statix-%: $(BUILDDIR)/test_statix_%

run-test-statix-%: $(BUILDDIR)/test_statix_%
	$(BUILDDIR)/test_statix_$*

clean-test-statix-%:
	$(DEL) $(BUILDDIR)/test_statix_$*

########################################################################

TESTS_XARRAY = \
	$(BUILDDIR)/test_xarray \
	$(BUILDDIR)/test_xarray_algebra \
	$(BUILDDIR)/test_xarray_structural

test-xarray: verb.build.test.xarray \
	$(TESTS_XARRAY) \
	verb.build.test.xarray.done

run-test-xarray: verb.run.test.xarray \
	run-test-xarray-run run-test-xarray_algebra-run run-test-xarray_structural-run \
	verb.run.test.xarray.done

run-test-xarray-run: $(BUILDDIR)/test_xarray
	$(BUILDDIR)/test_xarray

run-test-xarray_algebra-run: $(BUILDDIR)/test_xarray_algebra
	$(BUILDDIR)/test_xarray_algebra

run-test-xarray_structural-run: $(BUILDDIR)/test_xarray_structural
	$(BUILDDIR)/test_xarray_structural

clean-test-xarray:
	$(DEL) $(TESTS_XARRAY)

########################################################################

TESTS_TENSOR = \
	$(BUILDDIR)/test_tensor \
	$(BUILDDIR)/test_tensor_distance_l1

test-tensor: verb.build.test.tensor \
	$(TESTS_TENSOR) \
	verb.build.test.tensor.done

run-test-tensor: verb.run.test.tensor \
	run-test-tensor-run run-test-tensor_distance_l1-run \
	verb.run.test.tensor.done

run-test-tensor-run: $(BUILDDIR)/test_tensor
	$(BUILDDIR)/test_tensor

run-test-tensor_distance_l1-run: $(BUILDDIR)/test_tensor_distance_l1
	$(BUILDDIR)/test_tensor_distance_l1

clean-test-tensor:
	$(DEL) $(TESTS_TENSOR)

########################################################################

TESTS_MD = \
	$(BUILDDIR)/test_md_tensor \
	$(BUILDDIR)/test_md_math \
	$(BUILDDIR)/test_md_iterate \
	$(BUILDDIR)/test_md_cuda \
	$(BUILDDIR)/test_md_distance_l1

test-md: verb.build.test.md \
	$(TESTS_MD) \
	verb.build.test.md.done

run-test-md: verb.run.test.md \
	run-test-md_tensor-run run-test-md_math-run run-test-md_iterate-run run-test-md_cuda-run run-test-md_distance_l1-run \
	verb.run.test.md.done

run-test-md_tensor-run: $(BUILDDIR)/test_md_tensor
	$(BUILDDIR)/test_md_tensor

run-test-md_math-run: $(BUILDDIR)/test_md_math
	$(BUILDDIR)/test_md_math

run-test-md_iterate-run: $(BUILDDIR)/test_md_iterate
	$(BUILDDIR)/test_md_iterate

# md_cuda uses a malloc-backed fake CUDA runtime (tests/fakecuda) to exercise
# teeny/md/cuda.h structurally without a CUDA toolkit.
$(BUILDDIR)/test_md_cuda: tests/test_md_cuda.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Itests/fakecuda $(TESTFLAGS) -o $@ $<

run-test-md_cuda-run: $(BUILDDIR)/test_md_cuda
	$(BUILDDIR)/test_md_cuda

run-test-md_distance_l1-run: $(BUILDDIR)/test_md_distance_l1
	$(BUILDDIR)/test_md_distance_l1

clean-test-md:
	$(DEL) $(TESTS_MD)

########################################################################
# 	Messages
########################################################################

verb.build.test:
	$(call verb, "Building all tests...")

verb.build.test.done:
	$(call verb, "Building all tests: Done.")

verb.build.test.statix:
	$(call verb, "Building statix tests...")

verb.build.test.statix.done:
	$(call verb, "Building statix tests: Done.")

verb.run.test:
	$(call verb, "Running all tests...")

verb.run.test.done:
	$(call verb, "Running all tests: Done.")

verb.run.test.statix:
	$(call verb, "Running statix tests...")

verb.run.test.statix.done:
	$(call verb, "Running statix tests: Done.")

verb.build.test.xarray:
	$(call verb, "Building xarray tests...")

verb.build.test.xarray.done:
	$(call verb, "Building xarray tests: Done.")

verb.run.test.xarray:
	$(call verb, "Running xarray tests...")

verb.run.test.xarray.done:
	$(call verb, "Running xarray tests: Done.")

verb.build.test.tensor:
	$(call verb, "Building tensor tests...")

verb.build.test.tensor.done:
	$(call verb, "Building tensor tests: Done.")

verb.run.test.tensor:
	$(call verb, "Running tensor tests...")

verb.run.test.tensor.done:
	$(call verb, "Running tensor tests: Done.")

verb.build.test.md:
	$(call verb, "Building md tests...")

verb.build.test.md.done:
	$(call verb, "Building md tests: Done.")

verb.run.test.md:
	$(call verb, "Running md tests...")

verb.run.test.md.done:
	$(call verb, "Running md tests: Done.")

verb.clean.test:
	$(call verb, "Cleaning all tests...")

verb.clean.test.done:
	$(call verb, "Cleaning all tests: Done.")
