"""Double-talk detection: when the user speaks over the assistant.

If a filter adapts during double-talk it learns the user's voice as echo
and diverges. Rule here: freeze on classifier-DOUBLE hysteresis or on a
large residual-vs-reference ratio (Geigel-style: something other than
predictable echo is present). Otherwise adapt freely — echo-only stretches
are what converge the filter.

Deliberately NOT included: normalized mic-error cross-correlation
(MECC/XMCC-style). It was implemented, measured, and removed: under AEC3's
nonlinear suppression the residual carries no structure separating
double-talk from echo-only (both score high; the statistic needs
linear-filter error access this binding does not expose). Kept in the
record because the literature recommends it and the measurement says
otherwise for this backend. Conservative throughout: freezing too often
only slows convergence, while adapting through double-talk corrupts
the model.
"""
from __future__ import annotations


class DoubleTalkDetector:
    def __init__(self, residual_ratio: float = 0.5):
        self.residual_ratio = residual_ratio

    def reset(self) -> None:
        return None

    def observe(self, mic_vec, err_vec) -> float:
        """Accepted for interface compatibility; no internal state to update."""
        return 0.0

    def xi(self) -> float:
        """Correlation statistic placeholder (see module docstring)."""
        return 0.0

    def is_double_talk(self, mic_rms: float, ref_rms: float, res_rms: float,
                       last_label: str | None = None) -> bool:
        """True -> freeze coefficients this frame (keep filtering)."""
        if last_label == "DOUBLE":
            return True
        if ref_rms < 1e-7:
            return False
        return (res_rms * res_rms) / (ref_rms * ref_rms + 1e-8) > self.residual_ratio

    def should_adapt(self, mic_rms: float, ref_rms: float, res_rms: float,
                     last_label: str | None = None) -> bool:
        return not self.is_double_talk(mic_rms, ref_rms, res_rms, last_label)
