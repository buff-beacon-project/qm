#[macro_use]
extern crate criterion;
use num_complex::Complex;
use criterion::{black_box, Criterion};
use qm::fibc;

fn setup_benchmarks(c : &mut Criterion) {
  c.bench_function(
    "Fibonacci",
    |b| b.iter(|| {
      fibc(black_box(Complex{re:1.1, im:2.0}));
    })
  );
}

criterion_group!(benches, setup_benchmarks);
criterion_main!(benches);
