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
	verb.build.test.done

run-test: verb.run.test \
	run-test-statix \
	run-test-xarray \
	run-test-tensor \
	verb.run.test.done

clean-test: verb.clean.test \
	clean-test-statix \
	clean-test-xarray \
	clean-test-tensor \
	verb.clean.test.done

.PHONY: all clean test run-test clean-test
.PHONY: test-statix run-test-statix clean-test-statix
.PHONY: test-xarray run-test-xarray clean-test-xarray
.PHONY: test-tensor run-test-tensor clean-test-tensor

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
	$(BUILDDIR)/test_xarray_algebra

test-xarray: verb.build.test.xarray \
	$(TESTS_XARRAY) \
	verb.build.test.xarray.done

run-test-xarray: verb.run.test.xarray \
	run-test-xarray-run run-test-xarray_algebra-run \
	verb.run.test.xarray.done

run-test-xarray-run: $(BUILDDIR)/test_xarray
	$(BUILDDIR)/test_xarray

run-test-xarray_algebra-run: $(BUILDDIR)/test_xarray_algebra
	$(BUILDDIR)/test_xarray_algebra

clean-test-xarray:
	$(DEL) $(TESTS_XARRAY)

########################################################################

TESTS_TENSOR = \
	$(BUILDDIR)/test_tensor

test-tensor: verb.build.test.tensor \
	$(TESTS_TENSOR) \
	verb.build.test.tensor.done

run-test-tensor: verb.run.test.tensor \
	run-test-tensor-run \
	verb.run.test.tensor.done

run-test-tensor-run: $(BUILDDIR)/test_tensor
	$(BUILDDIR)/test_tensor

clean-test-tensor:
	$(DEL) $(TESTS_TENSOR)

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

verb.clean.test:
	$(call verb, "Cleaning all tests...")

verb.clean.test.done:
	$(call verb, "Cleaning all tests: Done.")
