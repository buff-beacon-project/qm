#[macro_use]
extern crate criterion;
use criterion::{black_box, Criterion};
use qm::fib;

fn setup_benchmarks(c : &mut Criterion) {
  c.bench_function(
    "Fibonacci",
    |b| b.iter(|| {
      fib(black_box(32))
    })
  );
}

criterion_group!(benches, setup_benchmarks);
criterion_main!(benches);
