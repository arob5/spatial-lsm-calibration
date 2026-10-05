"""Conjugacy: a variance or a covariance block integrated out of a model,
and its closed-form law given everything else.

Where this sits
---------------
::

    probability.model.FactoredDistribution.marginalize  -> a FactoredDistribution
    probability.posterior.Posterior.full_conditional    -> FullConditional
      (both read a ConjugateRule from this module)

Two rules are matched on the declarations, never on values. Both are
optional: a model with an inverse-gamma or inverse-Wishart component is
valid with those components as parameters, sampled jointly.

R1: an inverse gamma on a covariance scale
-------------------------------------------
It applies when a component :math:`v`, scalar or indexed by a dim
:math:`d`, has a factor reading no component whose law is an inverse gamma
:math:`\\mathrm{IG}(a, b)` (alone, under ``iid_over_dim`` or under
``independent_over_dim``), and exactly one part reads :math:`v`: a factor
whose law is a ``GaussianSpec`` with covariance ``ScaledSpec(base,
scale=v)`` or ``BlockDiagonalSpec(ScaledSpec(base, scale=v), by=g)``, ``g``
being :math:`d` when :math:`v` is indexed by it. For group :math:`s`, with
:math:`n_s` entries, residual :math:`r_s = y_s - m_s` and :math:`q_s =
r_s^\\top R_s^{-1} r_s` (:math:`R_s` the base over the group),

.. math::

    y_s \\sim t_{2a}\\big(m_s, (b/a) R_s\\big), \\qquad
    v_s \\mid \\text{rest} \\sim \\mathrm{IG}\\big(a + \\tfrac{n_s}{2},\\ b + \\tfrac{q_s}{2}\\big).

A scalar :math:`v` shared by every group gives one Student-t and one
inverse gamma, with :math:`n = \\sum_s n_s` and :math:`q = \\sum_s q_s`; a
label of :math:`d` that no group has keeps its prior.

R2: an inverse Wishart on a covariance block
---------------------------------------------
It applies when a :math:`p \\times p` component :math:`S` has a factor
reading no component whose law is
:math:`\\mathcal W^{-1}_p(\\nu, \\Psi)`, and exactly one part reads it: a
factor whose law is a ``GaussianSpec`` with covariance
``BlockDiagonalSpec(SubmatrixSpec(S, label_map=mu), by=g)``, nothing added,
whose every group maps to the same :math:`q` rows :math:`O` of :math:`S`.
With :math:`n` groups, residuals :math:`r_i` in :math:`O`'s order,
:math:`E = \\sum_i r_i r_i^\\top` and :math:`\\nu' = \\nu - (p - q)`, the
marginal is a matrix Student-t
(:mod:`~sipnet_calibration.probability.scale_mixtures`) and

.. math::

    S_{OO} \\mid \\text{rest} \\sim \\mathcal W^{-1}_q(\\nu' + n,\\ \\Psi_{OO} + E),

the rest of :math:`S` drawn from its prior given :math:`S_{OO}`: with
:math:`R` the other rows, :math:`S_{RR \\cdot O} \\sim \\mathcal
W^{-1}_{p-q}(\\nu, \\Psi_{RR \\cdot O})` and :math:`S_{OO}^{-1} S_{OR} \\mid
S_{RR \\cdot O} \\sim \\mathcal{MN}(\\Psi_{OO}^{-1}\\Psi_{OR},\\
\\Psi_{OO}^{-1},\\ S_{RR \\cdot O})`, independent of :math:`S_{OO}`.

R2 does not cover groups that map to different rows of :math:`S`, as when
sites' records are ragged, nor a term added to the submatrix; joint
sampling does.

Functions and classes
---------------------
:func:`marginalize`
    A bound model with components integrated out
    (:meth:`FactoredDistribution.marginalize
    <sipnet_calibration.probability.model.FactoredDistribution.marginalize>`).
:func:`conjugate_rule`, :class:`ConjugateRule`
    The rule that applies to a component, and its full conditional's
    parameters from residuals, which :meth:`Posterior.full_conditional
    <sipnet_calibration.probability.posterior.Posterior.full_conditional>`
    reads.
:class:`InverseWishartGivenRows`
    R2's full conditional when the groups observe some rows of the matrix.

Usage
-----
::

    marginal = model.marginalize(["leaf_area_index_error_scale"])
    scales = posterior.full_conditional("leaf_area_index_error_scale")
    evaluation = posterior.evaluate(theta)                # one simulator batch
    draw = scales.sample(key, evaluation)                 # IG(a + n_s/2, b + q_s/2)
    theta = posterior.theta_with(theta, {"leaf_area_index_error_scale": draw})
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import xarray as xr
from tensorflow_probability.substrates import jax as tfp

from sipnet_calibration.probability._validation import as_names, truncated
from sipnet_calibration.probability.covariance import BlockDiagonalSpec, ScaledSpec, SubmatrixSpec
from sipnet_calibration.probability.families import InverseWishart
from sipnet_calibration.probability.model import (
    FactoredDistribution,
    _covariance_scope,
    check_name_is_a_component,
    joint,
)
from sipnet_calibration.probability.parts import FactorSpec, GaussianSpec
from sipnet_calibration.probability.scale_mixtures import (
    MatrixStudentTSpec,
    StudentTSpec,
    inverse_wishart_log_prob,
    sample_inverse_wishart,
)
from sipnet_calibration.probability.spec import ArraySpec

__all__ = [
    "INVERSE_GAMMA_SCALE",
    "INVERSE_WISHART",
    "ConjugateRule",
    "InverseWishartGivenRows",
    "conjugate_rule",
    "marginalize",
]

tfd = tfp.distributions

Array = jax.Array

#: R1's name: an inverse gamma on a covariance scale.
INVERSE_GAMMA_SCALE = "inverse gamma scale"

#: R2's name: an inverse Wishart on a covariance block.
INVERSE_WISHART = "inverse Wishart"


def marginalize(model: FactoredDistribution, component_names: Sequence[str]) -> FactoredDistribution:
    """*model* with *component_names* integrated out, each by R1 or R2 in
    turn: its factor dropped, and the Gaussian factor reading it replaced by
    the marginal, a Student-t, in its place in declaration order. The
    result is bound at *model*'s labels and inputs.

    Raises
    ------
    TypeError
        If *component_names* is not a sequence of names.
    KeyError
        If a name is not a component of the model it is integrated out of.
    ValueError
        If *component_names* is empty, or no rule applies to a name; the
        message names the condition that failed.
    """
    names = as_names(component_names, message_name="component_names")
    check_names_are_given(names)
    for name in names:
        model = _marginal_model(model, conjugate_rule(model, name))
    return model


def conjugate_rule(model: FactoredDistribution, name: str) -> ConjugateRule:
    """The rule that applies to the component *name* of *model*: R1 or R2,
    matched on the declarations.

    Raises
    ------
    KeyError
        If *name* is not a component.
    ValueError
        If no rule applies, naming the condition that failed.
    """
    spec = model.spec
    check_name_is_a_component(name, spec)
    prior = spec._part_of(name)
    check_component_has_a_factor_of_its_own(name, prior)
    check_prior_reads_no_component(name, prior)
    law = model._factors[prior.name].law
    check_law_has_a_conjugate_rule(name, prior, law, model._layout.index_shape(name))
    if type(law) is InverseWishart:
        return _matrix_match(model, name, prior, law)
    return _scale_match(model, name, prior, law)


@dataclass(frozen=True, eq=False, kw_only=True)
class ConjugateRule:
    """A rule matched to a component (the module docstring's R1 or R2): its
    factor, the Gaussian factor reading it, the marginal law that factor
    becomes, and the full conditional's parameters from residuals. Made by
    :func:`conjugate_rule`. Compared and hashed by identity.

    Attributes
    ----------
    rule : str
        :data:`INVERSE_GAMMA_SCALE` or :data:`INVERSE_WISHART`.
    component : ArraySpec
        The component integrated out, or drawn.
    prior, reader : FactorSpec
        Its factor, and the Gaussian factor reading it.
    marginal_law : StudentTSpec or MatrixStudentTSpec
        The reader's law with the component integrated out.
    prior_shape, prior_scale : numpy.ndarray or None
        R1's :math:`a` and :math:`b` over the component's block.
    prior_law : InverseWishart or None
        R2's prior.
    label_map_name : str or None
        R2's label map, from the reader's entries to the matrix's rows.
    """

    rule: str
    component: ArraySpec
    prior: FactorSpec
    reader: FactorSpec
    marginal_law: StudentTSpec | MatrixStudentTSpec
    prior_shape: Any = None
    prior_scale: Any = None
    prior_law: Any = None
    label_map_name: str | None = None

    @property
    def residual_names(self) -> tuple[str, str]:
        """The reader's event and its mean, whose difference the residuals
        are."""
        return self.reader.event[0].name, self.reader.law.mean[0]

    def covariance_names(self, model: FactoredDistribution) -> tuple[str, ...]:
        """The components and inputs the marginal's covariance reads."""
        mean = self.reader.law.mean[0]
        return tuple(n for n in self.marginal_law.reads if n != mean and n in model.spec)

    def structure(self, model: FactoredDistribution) -> Any:
        """The marginal law bound at the reader's labels in *model*: its
        groups and their covariances."""
        reader = model._factors[self.reader.name]
        return self.marginal_law._at(_covariance_scope(model, self.reader, reader.fixed_reads))

    def conditional_parameters(
        self, model: FactoredDistribution, structure: Any, residual: Array, covariance_values: Mapping[str, Array]
    ) -> dict[str, Array]:
        """The full conditional's parameters per sample, ``{name: (J,
        ...)}``, from the residuals ``(J, n)`` and what the covariance reads,
        each ``(J, *block)``: R1's shape and scale per label, :math:`a +
        n_s/2` and :math:`b + q_s/2`; R2's degrees of freedom :math:`\\nu' +
        n` and scale :math:`\\Psi_{OO} + E`."""
        if self.rule == INVERSE_WISHART:
            grouped = residual[:, structure.entries]
            scatter = jnp.einsum("jgi,jgk->jik", grouped, grouped)
            n, _ = structure.entries.shape
            return {
                "degrees_of_freedom": jnp.full((residual.shape[0],), structure.degrees_of_freedom + n),
                "scale": structure.scale + scatter,
            }
        fixed = model._factors[self.reader.name].fixed_reads

        def quadratic_forms(residual_j: Array, values_j: Mapping[str, Array]) -> Array:
            reads = {**values_j, **fixed}
            forms = []
            for group in structure.groups:
                operator = group.covariance.operator(structure.reads_of_group(reads, group))
                forms.append(jnp.sum(operator.whiten(residual_j[group.positions]) ** 2))
            return jnp.stack(forms)

        forms = jax.vmap(quadratic_forms)(residual, dict(covariance_values))
        counts = np.asarray([len(group.positions) for group in structure.groups], dtype=np.float64)
        if not self.component.indexed_by:
            return {"shape": jnp.full(forms.shape[:1], self.prior_shape + counts.sum() / 2.0),
                    "scale": self.prior_scale + forms.sum(axis=1) / 2.0}
        (dim,) = self.component.indexed_by
        labels = model.coords[dim]
        places = np.asarray([labels.get_loc(group.label) for group in structure.groups], dtype=np.int64)
        shape = self.prior_shape + np.bincount(places, weights=counts, minlength=len(labels)) / 2.0
        scale = jnp.asarray(self.prior_scale) + jnp.zeros((residual.shape[0], len(labels))).at[:, places].add(forms / 2.0)
        return {"shape": jnp.broadcast_to(jnp.asarray(shape), scale.shape), "scale": scale}

    def conditional_law(self, model: FactoredDistribution, structure: Any, parameters: Mapping[str, Array]) -> Any:
        """The full conditional at one sample's *parameters*: an inverse
        gamma per label (under ``tfd.Independent`` when indexed); an
        :class:`~sipnet_calibration.probability.families.InverseWishart`
        when the groups observe every row of the matrix, else an
        :class:`InverseWishartGivenRows`."""
        if self.rule == INVERSE_GAMMA_SCALE:
            law = tfd.InverseGamma(parameters["shape"], parameters["scale"])
            return law if not self.component.indexed_by else tfd.Independent(law, 1)
        rows, rest = self._rows_and_rest(model, structure)
        if not len(rest):
            return InverseWishart(parameters["degrees_of_freedom"], parameters["scale"])
        return InverseWishartGivenRows(
            degrees_of_freedom=self.prior_law.degrees_of_freedom, scale=self.prior_law.scale, rows=rows,
            observed_degrees_of_freedom=parameters["degrees_of_freedom"], observed_scale=parameters["scale"],
        )

    def conditional_sample(self, key: Array, model: FactoredDistribution, structure: Any, parameters: Mapping[str, Array]) -> Array:
        """One draw of the full conditional per sample of *parameters*,
        ``(J, *block)``."""
        if self.rule == INVERSE_GAMMA_SCALE:
            return tfd.InverseGamma(parameters["shape"], parameters["scale"]).sample(seed=key)
        rows, rest = self._rows_and_rest(model, structure)
        matrix_key, rest_key = jax.random.split(key)
        observed = sample_inverse_wishart(matrix_key, parameters["degrees_of_freedom"], parameters["scale"])
        if not len(rest):
            return observed
        return _completed_from_the_prior(
            rest_key, observed, self.prior_law.degrees_of_freedom, self.prior_law.scale, rows, rest
        )

    def _rows_and_rest(self, model: FactoredDistribution, structure: Any) -> tuple[np.ndarray, np.ndarray]:
        """R2's rows the groups observe, :math:`O`, ascending, and the
        others."""
        p = self.prior_law.scale.shape[-1]
        positions = np.asarray(model._factors[self.reader.name].fixed_reads[self.label_map_name])
        rows = np.sort(positions[structure.entries[0]])
        return rows, np.setdiff1d(np.arange(p), rows)


@dataclass(frozen=True, eq=False, kw_only=True)
class InverseWishartGivenRows:
    """A law over a :math:`p \\times p` matrix :math:`S`: the rows
    :math:`O` from :math:`\\mathcal W^{-1}_q(\\nu_O, \\Psi_O)`, the rest from
    :math:`\\mathcal W^{-1}_p(\\nu, \\Psi)` given them,

    .. math::

        \\log p(S) = \\log \\mathcal W^{-1}_q(S_{OO}; \\nu_O, \\Psi_O)
            + \\log \\mathcal W^{-1}_p(S; \\nu, \\Psi)
            - \\log \\mathcal W^{-1}_q(S_{OO}; \\nu - (p - q), \\Psi_{OO}),

    against Lebesgue measure on the lower triangle: R2's full conditional
    when the groups observe :math:`q < p` rows. Compared and hashed by
    identity.

    Attributes
    ----------
    degrees_of_freedom, scale
        :math:`\\nu` and :math:`\\Psi`, the prior's.
    rows : numpy.ndarray
        :math:`O`, ascending positions.
    observed_degrees_of_freedom, observed_scale
        :math:`\\nu_O` and :math:`\\Psi_O`, ``(q, q)``.
    """

    degrees_of_freedom: Any
    scale: Any
    rows: np.ndarray
    observed_degrees_of_freedom: Any
    observed_scale: Any

    @property
    def event_shape(self) -> tuple[int, ...]:
        p = self.scale.shape[-1]
        return (p, p)

    @property
    def dtype(self) -> Any:
        return jnp.float64

    def log_prob(self, value: Any) -> Array:
        """The log density at *value*, ``(..., p, p) -> (...)``."""
        value = jnp.asarray(value, dtype=jnp.float64)
        rows = self.rows
        p, q = self.scale.shape[-1], len(rows)
        observed = value[..., rows[:, None], rows[None, :]]
        prior_observed = jnp.asarray(self.scale)[np.ix_(rows, rows)]
        return (
            inverse_wishart_log_prob(observed, self.observed_degrees_of_freedom, self.observed_scale)
            + inverse_wishart_log_prob(value, self.degrees_of_freedom, self.scale)
            - inverse_wishart_log_prob(observed, self.degrees_of_freedom - (p - q), prior_observed)
        )

    def sample(self, sample_shape: tuple[int, ...] = (), seed: Array | None = None) -> Array:
        """Draws of :math:`S`, ``(*sample_shape, p, p)``: :math:`S_{OO}`,
        then the rest given it.

        Raises
        ------
        TypeError
            If *seed* is not given.
        """
        check_seed_is_given(seed)
        sample_shape = tuple(sample_shape) if isinstance(sample_shape, (tuple, list)) else (int(sample_shape),)
        observed_key, rest_key = jax.random.split(seed)
        observed = sample_inverse_wishart(observed_key, self.observed_degrees_of_freedom, self.observed_scale, sample_shape)
        rest = np.setdiff1d(np.arange(self.scale.shape[-1]), self.rows)
        return _completed_from_the_prior(rest_key, observed, self.degrees_of_freedom, self.scale, self.rows, rest)


# ── private: matching the rules ───────────────────────────────────────────────


def _scale_match(model: FactoredDistribution, name: str, prior: FactorSpec, law: Any) -> ConjugateRule:
    """R1 matched to *name*, or the condition that failed."""
    (component,) = prior.event
    check_scale_is_indexed_by_one_dim_at_most(component)
    reader = _the_gaussian_reader(model, name)
    covariance = reader.law.covariance
    grouping = covariance.by if isinstance(covariance, BlockDiagonalSpec) else None
    scaled = covariance.block if isinstance(covariance, BlockDiagonalSpec) else covariance
    check_covariance_is_scaled_by(reader, name, scaled)
    # A scale on d inside a grouping by anything but d is not one number per
    # group, which binding the model already refused (ScaledSpec).
    shape, scale = _inverse_gamma_parameters(law, model._layout.index_shape(name))
    mean = reader.law.mean[0]
    if component.indexed_by:
        (dim,) = component.indexed_by
        labels = model.coords[dim]
        marginal = StudentTSpec(
            mean=mean, covariance=scaled.base, by=dim,
            shape=xr.DataArray(shape, dims=(dim,), coords={dim: labels}),
            scale=xr.DataArray(scale, dims=(dim,), coords={dim: labels}),
        )
    else:
        base = scaled.base if grouping is None else BlockDiagonalSpec(scaled.base, by=grouping)
        marginal = StudentTSpec(mean=mean, covariance=base, shape=float(shape), scale=float(scale))
    return ConjugateRule(
        rule=INVERSE_GAMMA_SCALE, component=component, prior=prior, reader=reader, marginal_law=marginal,
        prior_shape=np.asarray(shape), prior_scale=np.asarray(scale),
    )


def _matrix_match(model: FactoredDistribution, name: str, prior: FactorSpec, law: InverseWishart) -> ConjugateRule:
    """R2 matched to *name*, or the condition that failed."""
    (component,) = prior.event
    reader = _the_gaussian_reader(model, name)
    covariance = reader.law.covariance
    check_covariance_is_a_submatrix_per_group(reader, name, covariance)
    rows, columns = component.element_axes
    scale = xr.DataArray(
        np.asarray(law.scale), dims=(rows, columns),
        coords={rows: component.element_axes[rows], columns: component.element_axes[columns]},
    )
    marginal = MatrixStudentTSpec(
        mean=reader.law.mean[0], by=covariance.by, label_map=reader.label_maps[covariance.block.label_map],
        degrees_of_freedom=float(law.degrees_of_freedom), scale=scale,
    )
    # Bound here so that groups mapping to different rows are refused in the
    # rule's words, before anything is built.
    marginal._at(_covariance_scope(model, reader, model._factors[reader.name].fixed_reads))
    return ConjugateRule(
        rule=INVERSE_WISHART, component=component, prior=prior, reader=reader, marginal_law=marginal,
        prior_law=law, label_map_name=covariance.block.label_map,
    )


def _the_gaussian_reader(model: FactoredDistribution, name: str) -> FactorSpec:
    """The one part reading *name*, a Gaussian factor of which it is not the
    mean."""
    readers = [p for p in model.spec.parts if name in p.given]
    check_component_has_one_reader(name, readers)
    (reader,) = readers
    check_reader_is_a_gaussian_factor(name, reader)
    check_component_is_not_the_mean(name, reader)
    return reader


def _inverse_gamma_parameters(law: Any, index_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    """An inverse gamma's shape and scale over the block, ``index_shape``
    each: alone, under ``tfd.Sample`` (``iid_over_dim``) or under
    ``tfd.Independent`` (``independent_over_dim``); ``None`` for any other
    law."""
    inner = law.distribution if type(law) in (tfd.Sample, tfd.Independent) else law
    if type(inner) is not tfd.InverseGamma:
        return None
    shape = np.broadcast_to(np.asarray(inner.concentration, dtype=np.float64), index_shape)
    scale = np.broadcast_to(np.asarray(inner.scale, dtype=np.float64), index_shape)
    return shape, scale


def _marginal_model(model: FactoredDistribution, match: ConjugateRule) -> FactoredDistribution:
    """*model* with the matched component's factor dropped and its reader's
    law replaced by the marginal, bound at the same labels and inputs."""
    spec = model.spec
    reader = match.reader
    kept = set(match.marginal_law.reads)
    constants = {n: c for n, c in reader.constants.items() if n in kept}
    marginal = FactorSpec(
        reader.event,
        law=match.marginal_law,
        constants=constants,
        label_maps={n: m for n, m in reader.label_maps.items() if n in kept},
        own_dims=[d for d in reader.own_dims if any(d in c.dims for c in constants.values())],
        provenance=_provenance(match),
    )
    parts = [marginal if part is reader else part for part in spec.parts if part is not match.prior]
    return joint(*parts, inputs=spec.inputs).bind(coords=dict(model.coords), inputs=dict(model.inputs))


def _provenance(match: ConjugateRule) -> str | None:
    """The marginal factor's provenance: its reader's, and its prior's
    under the rule."""
    reader, prior = match.reader.provenance, match.prior.provenance
    if reader is None and prior is None:
        return None
    return (
        f"{reader or 'No provenance given.'} {match.component.name!r} integrated out by the {match.rule} rule, "
        f"its prior: {prior or 'no provenance given.'}"
    )


def _completed_from_the_prior(
    key: Array, observed: Array, degrees_of_freedom: Any, scale: Any, rows: np.ndarray, rest: np.ndarray
) -> Array:
    """Matrices :math:`S`, ``(..., p, p)``, with :math:`S_{OO}` *observed*
    and the rest drawn from :math:`\\mathcal W^{-1}_p(\\nu, \\Psi)` given it
    (the module docstring)."""
    scale = jnp.asarray(scale, dtype=jnp.float64)
    lead = observed.shape[:-2]
    q, r = len(rows), len(rest)
    psi_oo = scale[np.ix_(rows, rows)]
    psi_or = scale[np.ix_(rows, rest)]
    psi_rr = scale[np.ix_(rest, rest)]
    psi_oo_lower = jnp.linalg.cholesky(psi_oo)
    coefficients = jax.scipy.linalg.cho_solve((psi_oo_lower, True), psi_or)
    conditional_scale = psi_rr - psi_or.T @ coefficients
    conditional_key, noise_key = jax.random.split(key)
    conditional = sample_inverse_wishart(conditional_key, degrees_of_freedom, (conditional_scale + conditional_scale.T) / 2.0, lead)
    # The row covariance is the inverse of Psi_OO, whose factor is the
    # inverse transpose of Psi_OO's: A A^T = Psi_OO^-1 for A = L^-T.
    row_factor = jax.scipy.linalg.solve_triangular(psi_oo_lower, jnp.eye(q), lower=True, trans=1)
    column_factor = jnp.linalg.cholesky(conditional)
    noise = jax.random.normal(noise_key, (*lead, q, r), dtype=jnp.float64)
    regression = coefficients + row_factor @ noise @ jnp.swapaxes(column_factor, -1, -2)
    cross = observed @ regression
    remainder = conditional + jnp.swapaxes(regression, -1, -2) @ observed @ regression
    p = q + r
    matrix = jnp.zeros((*lead, p, p), dtype=jnp.float64)
    matrix = matrix.at[..., rows[:, None], rows[None, :]].set(observed)
    matrix = matrix.at[..., rows[:, None], rest[None, :]].set(cross)
    matrix = matrix.at[..., rest[:, None], rows[None, :]].set(jnp.swapaxes(cross, -1, -2))
    matrix = matrix.at[..., rest[:, None], rest[None, :]].set(remainder)
    return (matrix + jnp.swapaxes(matrix, -1, -2)) / 2.0


# ── checks ────────────────────────────────────────────────────────────────────


def check_names_are_given(names: Sequence[str]) -> None:
    """Something is integrated out."""
    if not names:
        raise ValueError("component_names is empty; name the components to integrate out.")


def check_component_has_a_factor_of_its_own(name: str, part: Any) -> None:
    """A conjugate component is declared alone by a factor, whose law is its
    prior."""
    if not isinstance(part, FactorSpec) or len(part.event) != 1:
        what = part.name if isinstance(part, FactorSpec) else f"the {type(part).__name__} {part.name!r}"
        raise ValueError(
            f"no conjugate rule applies to {name!r}: it is declared by {what!r}, not by a factor of its own; a rule "
            "integrates out a component whose factor declares it alone."
        )


def check_prior_reads_no_component(name: str, prior: FactorSpec) -> None:
    """A conjugate component's prior reads no component, so its parameters
    are fixed numbers."""
    if prior.given:
        raise ValueError(
            f"no conjugate rule applies to {name!r}: its law reads {truncated(list(prior.given))}, and a rule needs "
            "a prior of fixed numbers; give its shape and scale as numbers or constants."
        )


def check_law_has_a_conjugate_rule(name: str, prior: FactorSpec, law: Any, index_shape: tuple[int, ...]) -> None:
    """A conjugate component's law is an inverse gamma or an inverse
    Wishart."""
    if type(law) is InverseWishart or _inverse_gamma_parameters(law, index_shape) is not None:
        return
    raise ValueError(
        f"no conjugate rule applies to {name!r}: its law is {prior.law_name!r}, neither an inverse gamma (on a "
        "covariance scale) nor an inverse Wishart (on a covariance block)."
    )


def check_scale_is_indexed_by_one_dim_at_most(component: ArraySpec) -> None:
    """A covariance scale is a scalar, or one per label of one dim."""
    if len(component.indexed_by) > 1 or component.element_axes:
        raise ValueError(
            f"no conjugate rule applies to {component.name!r}: it is indexed by {component.indexed_by} with element "
            f"axes {list(component.element_axes)}, and R1 integrates out a scalar scale or one per label of one dim."
        )


def check_component_has_one_reader(name: str, readers: Sequence[Any]) -> None:
    """A conjugate component is read by exactly one part."""
    if len(readers) != 1:
        found = [p.name for p in readers]
        raise ValueError(
            f"no conjugate rule applies to {name!r}: it is read by {len(readers)} parts {truncated(found)}, and a "
            "rule needs exactly one, a Gaussian factor's covariance."
        )


def check_reader_is_a_gaussian_factor(name: str, reader: Any) -> None:
    """A conjugate component's one reader is a factor with a GaussianSpec."""
    if not (isinstance(reader, FactorSpec) and isinstance(reader.law, GaussianSpec)):
        raise ValueError(
            f"no conjugate rule applies to {name!r}: it is read by {reader.name!r}, whose law is "
            f"{reader.law_name!r}, not a GaussianSpec."
        )


def check_component_is_not_the_mean(name: str, reader: FactorSpec) -> None:
    """A conjugate component is read by its Gaussian's covariance, not as its
    mean."""
    if name in reader.law.mean:
        raise ValueError(f"no conjugate rule applies to {name!r}: {reader.name!r} is centered on it.")


def check_covariance_is_scaled_by(reader: FactorSpec, name: str, scaled: Any) -> None:
    """R1's Gaussian covariance is ``name`` times a base, alone or per
    group."""
    if not (isinstance(scaled, ScaledSpec) and scaled.scale == name):
        raise ValueError(
            f"no conjugate rule applies to {name!r}: the covariance of {reader.name!r} is "
            f"{reader.law.covariance!r}, and R1 needs ScaledSpec(base, scale={name!r}), alone or as "
            f"BlockDiagonalSpec(ScaledSpec(base, scale={name!r}), by=...)."
        )


def check_covariance_is_a_submatrix_per_group(reader: FactorSpec, name: str, covariance: Any) -> None:
    """R2's Gaussian covariance is the bare submatrix of ``name`` per group,
    nothing added."""
    block = covariance.block if isinstance(covariance, BlockDiagonalSpec) else None
    if not (isinstance(block, SubmatrixSpec) and block.component == name):
        raise ValueError(
            f"no conjugate rule applies to {name!r}: the covariance of {reader.name!r} is {covariance!r}, and R2 "
            f"needs BlockDiagonalSpec(SubmatrixSpec({name!r}, label_map=...), by=...) with nothing added."
        )


def check_seed_is_given(seed: Any) -> None:
    """A draw is made from a key."""
    if seed is None:
        raise TypeError("an InverseWishartGivenRows draws from a key; give seed=jax.random.key(...).")
