.PHONY: run

libtfdir = tensorflow/bazel-bin/tensorflow/compiler/xla/xla_rs
libtf = $(libtfdir)/libxla_rs.so

$(libtf): xla_rs/xla_rs.cc xla_rs/xla_rs.h
	rm -Rf tensorflow/tensorflow/compiler/xla/xla_rs
	cp -Rf xla_rs tensorflow/tensorflow/compiler/xla/
	cd tensorflow && \
	    bazel build --define framework_shared_object=false tensorflow/compiler/xla/xla_rs:libxla_rs.so -j 2

run: $(libtf)
	LD_LIBRARY_PATH=$(libtfdir):$(LD_LIBRARY_PATH) LIBRARY_PATH=$(libtfdir):$(LIBRARY_PATH) cargo run --example basics
