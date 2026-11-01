using RustCall

# Populate RustCall's cache; @rust_crate also rebuilds when its inputs change.
RustCall.load_crate_bindings(joinpath(@__DIR__, "rust"))
