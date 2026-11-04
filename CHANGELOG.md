# Changelog

Notable changes to ProbingEnvironments. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **`prove` shows that a probe catches the bug it is named after.** It runs a
  probe healthy and with a planted fault, through any agent, and requires the
  first to pass and the second to be caught. This release carries one probe
  (`probes.discounting`), one fault (`faults.no_discount`, a setting, so the
  library needs no edit) and an `adaptors.sb3.SB3` agent.
- **Provisional:** `runner.provisional` holds only what the candidate rules
  share, and it will be replaced. Do not treat its results as stable yet.

### Internal

- Characterization pins (`tests/characterization/`) record how every
  environment behaves today, through the API users and Ajax rely on. They
  keep the 0.7 refactor honest: a restructuring must leave them unchanged,
  and each deliberate behaviour change shows up as a pin difference with its
  own upgrade note here. No change to the package.

## [0.6.0]

### Upgrading from 0.5.1

A minor release, because these changes break code written for 0.5.1:

- **gymnax 1.0 or newer is required.** 0.5.1 required gymnax 0.0.9, so it
  could not be installed next to gymnax 1.x, and its environments failed on
  their first step under gymnax 1.0. Upgrade gymnax to `>=1.0`.
- **The gymnax environments follow the gymnax 1.0 API.**
  - `env.step` returns six values,
    `(obs, state, reward, terminated, truncated, info)`, instead of five.
    Unpack the extra value, and bootstrap your value targets at `truncated`
    steps: a time limit is no longer folded into `done`.
  - Environments implement `is_terminated(state, params)` instead of
    `is_terminal`. Rename it in your subclasses.
  - `get_obs` takes `(state, params=None, key=None)`. Add the two optional
    arguments to your overrides.
- **The gymnax `TwoChoiceMDP` never terminates.** Its episodes now end only
  at the time limit, reported as `truncated`, so an agent should bootstrap
  there.
- **pytest is no longer installed with this package.** It moved to the
  development dependencies; add it to your own if you relied on it.
- **The Gymnasium environments draw from their own generator.**
  `reset(seed=s)` seeds `env.np_random`, as Gymnasium specifies, instead of
  reseeding numpy's global generator, and a later `reset()` continues that
  stream. Seeded start observations no longer match 0.5.1's, where only the
  first one depended on the seed. If your code relied on `reset` seeding
  `np.random`, seed it yourself.

### Added

- The `sb3` extra: `pip install "probing-environments[sb3]"` installs
  Stable-Baselines3 for the premade adaptor.
- A documentation site, versioned per release:
  https://yannberthelot.github.io/ProbingEnvironments/

### Changed

- `jax`, `flax` and `numpy` are declared as dependencies. The package
  imports them directly; before, they only arrived through gymnax.
- `six` is declared: gymnax 1.0.1 imports it without declaring it, so a fresh
  install could not import the package.
- Python 3.11 to 3.14 is supported; gymnax 1.0.1 supports Python below 3.15.
- The package is built with hatchling and versioned from git tags. Release
  tags are now named `vX.Y.Z`; earlier ones had no `v`.

### Deprecated

- `probing_environments.adaptors.ajax` warns on import and will be removed in
  0.7.0. Ajax ships its own probe adaptor in its test suite.

### Fixed

- A seeded run through the SB3 adaptor is reproducible. The Gymnasium
  environments reseeded numpy's global generator from OS entropy on every
  automatic reset, so each run differed and the README's PPO example failed
  about once in 50 runs.
