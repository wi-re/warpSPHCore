import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import FeatureCard from '@site/src/components/FeatureCard';

export default () => (
  <Layout
    title="warpSPHCore — operator wiki"
    description="Warp- and PyTorch-based core SPH operators: compact-hash neighbor search, differentiable density/gradient/divergence/curl/Laplacian/interpolation, CRK corrections, forward-mode JVP."
  >
    <header className="hero hero--primary">
      <div className="container text--center">
        <h1>
          warpSPHCore
          <span className="heroSubtitle">operator wiki</span>
        </h1>
        <p className="heroTagline">
          GPU-accelerated, differentiable SPH core operators — one page per
          operator: the equations, the options, the JVP variants, and the test
          files that pin every claim.
        </p>
        <div className="row">
          <div className="col">
            <Link
              className="button button--primary button--lg"
              to="docs"
            >
              Browse the wiki →
            </Link>
          </div>
          <div className="col">
            <Link
              className="button button--secondary button--lg"
              to="https://github.com/wi-re/warpSPHCore"
            >
              GitHub
            </Link>
          </div>
        </div>
      </div>
    </header>

    <main>
      <div className="container pad-top--lg pad-bottom--lg">
        <div className="row">
          <FeatureCard title="One page per operator">
            Density, gradient, divergence, curl, Laplacian, interpolation and
            covariance — the per-pair accumulation as implemented in the Warp
            kernels, every option, the support/gradient schemes, and the
            corrections each one accepts.
          </FeatureCard>
          <FeatureCard title="CRK corrections">
            Moment matrices, apparent volume/area, and the A/B correction
            tensors that lift the operators to corrected (linear-reproducing)
            consistency — with the analytic JVPs.
          </FeatureCard>
          <FeatureCard title="Differentiable end-to-end">
            Warp kernels wrapped as PyTorch functions: reverse-mode
            gradients and forward-mode JVP (value and geometry duals) with
            no host-device synchronisation on the hot path.
          </FeatureCard>
        </div>
      </div>
    </main>
  </Layout>
);
