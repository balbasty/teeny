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
	verb.build.test.done

run-test: verb.run.test \
	run-test-statix \
	run-test-xarray \
	verb.run.test.done

clean-test: verb.clean.test \
	clean-test-statix \
	clean-test-xarray \
	verb.clean.test.done

.PHONY: all clean test run-test clean-test
.PHONY: test-statix run-test-statix clean-test-statix
.PHONY: test-xarray run-test-xarray clean-test-xarray

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
	$(BUILDDIR)/test_xarray

test-xarray: verb.build.test.xarray \
	$(TESTS_XARRAY) \
	verb.build.test.xarray.done

run-test-xarray: verb.run.test.xarray \
	run-test-xarray-run \
	verb.run.test.xarray.done

run-test-xarray-run: $(BUILDDIR)/test_xarray
	$(BUILDDIR)/test_xarray

clean-test-xarray:
	$(DEL) $(TESTS_XARRAY)

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

verb.clean.test:
	$(call verb, "Cleaning all tests...")

verb.clean.test.done:
	$(call verb, "Cleaning all tests: Done.")
