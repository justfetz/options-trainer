"""The engine's specification.

Two kinds of test, and the distinction is the first lesson in evals:

  * Golden cases: Hull's worked examples, transcribed by hand into evals/hull_examples.json.
    Ground truth is the book. If the engine disagrees with Hull, the engine is wrong.
  * Property tests: claims the book makes in words ("gamma is greatest at the money near
    expiry"). Ground truth is the shape, swept across inputs. These catch sign errors and
    swapped arguments that a single golden case can miss.

Everything here is deterministic, so these are ordinary pytest assertions. The evals in
evals/run.py are the non-deterministic cousin: same idea, scored as a rate.
"""
import json
import math
from pathlib import Path

import pytest

from trainer import engine

CASES = {c["id"]: c for c in json.loads((Path(__file__).parent.parent / "evals" / "hull_examples.json").read_text())["cases"]
         if not c["id"].startswith("TODO")}


# ----------------------------------------------------------------------------- golden: Hull

def test_hull_15_6_d1_d2():
    c = CASES["hull-15.6-call"]
    i = c["inputs"]
    d1, d2 = engine.d1_d2(i["S"], i["K"], i["T"], i["r"], i["sigma"])
    assert d1 == pytest.approx(c["expect"]["d1"], abs=c["tolerance"]["d1"])
    assert d2 == pytest.approx(c["expect"]["d2"], abs=c["tolerance"]["d2"])


@pytest.mark.parametrize("case_id", ["hull-15.6-call", "hull-15.6-put"])
def test_hull_15_6_price(case_id):
    c = CASES[case_id]
    i = c["inputs"]
    price = engine.bs_price(i["S"], i["K"], i["T"], i["r"], i["sigma"], i["kind"])
    assert price == pytest.approx(c["expect"]["price"], abs=c["tolerance"]["price"])


def test_hull_15_9_dividends():
    c = CASES["hull-15.9-call-with-dividends"]
    i = c["inputs"]
    divs = [tuple(d) for d in i["dividends"]]
    assert engine.pv_dividends(divs, i["r"]) == pytest.approx(c["expect"]["pv_dividends"], abs=c["tolerance"]["pv_dividends"])
    price = engine.bs_price_with_dividends(i["S"], i["K"], i["T"], i["r"], i["sigma"], i["kind"], divs)
    assert price == pytest.approx(c["expect"]["price"], abs=c["tolerance"]["price"])


# ----------------------------------------------------------------------------- properties: Hull in words

def test_put_call_parity_holds_everywhere():
    """Hull (11.6): c + K e^{-rT} = p + S, for every European call/put pair."""
    for S in (30, 40, 50):
        for T in (0.1, 0.5, 2.0):
            for sigma in (0.1, 0.3, 0.8):
                c = engine.bs_price(S, 40, T, 0.05, sigma, "call")
                p = engine.bs_price(S, 40, T, 0.05, sigma, "put")
                assert abs(engine.put_call_parity_gap(c, p, S, 40, T, 0.05)) < 1e-9


def test_price_bounds_hull_ch11():
    """Hull 11.3: max(S - K e^{-rT}, 0) <= c <= S; max(K e^{-rT} - S, 0) <= p <= K e^{-rT}."""
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
    c = engine.bs_price(S, K, T, r, sigma, "call")
    p = engine.bs_price(S, K, T, r, sigma, "put")
    assert max(S - K * math.exp(-r * T), 0) <= c <= S
    assert max(K * math.exp(-r * T) - S, 0) <= p <= K * math.exp(-r * T)


def test_expiry_returns_intrinsic():
    assert engine.bs_price(42, 40, 0.0, 0.1, 0.2, "call") == pytest.approx(2.0)
    assert engine.bs_price(42, 40, 0.0, 0.1, 0.2, "put") == pytest.approx(0.0)


def test_delta_limits_hull_19_4():
    """Call delta -> 0 deep OTM, -> 1 deep ITM; put delta is call delta - 1 (no dividends)."""
    g_otm = engine.greeks(20, 40, 0.5, 0.05, 0.2, "call")
    g_itm = engine.greeks(80, 40, 0.5, 0.05, 0.2, "call")
    assert g_otm.delta < 0.01 and g_itm.delta > 0.99
    gc = engine.greeks(42, 40, 0.5, 0.1, 0.2, "call")
    gp = engine.greeks(42, 40, 0.5, 0.1, 0.2, "put")
    assert gp.delta == pytest.approx(gc.delta - 1.0, abs=1e-12)


def test_gamma_peaks_near_the_money_and_sharpens_near_expiry():
    """Hull 19.6: gamma is greatest for at-the-money options and increases as expiry approaches."""
    strikes = [30, 35, 38, 40, 42, 45, 50]
    gammas = [engine.greeks(40, K, 0.25, 0.05, 0.2, "call").gamma for K in strikes]
    assert strikes[gammas.index(max(gammas))] in (38, 40, 42)
    atm_far = engine.greeks(40, 40, 1.0, 0.05, 0.2, "call").gamma
    atm_near = engine.greeks(40, 40, 0.05, 0.05, 0.2, "call").gamma
    assert atm_near > atm_far


def test_gamma_and_vega_are_the_same_for_call_and_put():
    gc = engine.greeks(42, 40, 0.5, 0.1, 0.2, "call")
    gp = engine.greeks(42, 40, 0.5, 0.1, 0.2, "put")
    assert gc.gamma == pytest.approx(gp.gamma, abs=1e-12)
    assert gc.vega == pytest.approx(gp.vega, abs=1e-12)


def test_theta_sign_hull_19_5():
    """Theta is usually negative for a long option. Hull's exception: a deep in-the-money
    European put can have POSITIVE theta (you are waiting to receive K, and waiting costs interest).
    This is the case most people get wrong; it is in the golden set on purpose."""
    assert engine.greeks(40, 40, 0.5, 0.05, 0.2, "call").theta < 0
    assert engine.greeks(40, 40, 0.5, 0.05, 0.2, "put").theta < 0
    assert engine.greeks(10, 40, 0.5, 0.10, 0.2, "put").theta > 0


def test_greeks_match_finite_differences():
    """The analytic Greeks must agree with bumping the price. If they don't, one of them lies."""
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
    g = engine.greeks(S, K, T, r, sigma, "call")
    h = 1e-4
    f = lambda s=S, t=T, v=sigma, rr=r: engine.bs_price(s, K, t, rr, v, "call")
    assert g.delta == pytest.approx((f(s=S + h) - f(s=S - h)) / (2 * h), rel=1e-5)
    assert g.gamma == pytest.approx((f(s=S + h) - 2 * f() + f(s=S - h)) / h ** 2, rel=1e-3)
    assert g.vega == pytest.approx((f(v=sigma + h) - f(v=sigma - h)) / (2 * h), rel=1e-5)
    assert g.theta == pytest.approx(-(f(t=T + h) - f(t=T - h)) / (2 * h), rel=1e-4)
    assert g.rho == pytest.approx((f(rr=r + h) - f(rr=r - h)) / (2 * h), rel=1e-5)


def test_implied_vol_round_trips():
    for sigma in (0.08, 0.2, 0.45, 1.2):
        price = engine.bs_price(42, 40, 0.5, 0.1, sigma, "call")
        assert engine.implied_vol(price, 42, 40, 0.5, 0.1, "call") == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    with pytest.raises(engine.EngineError):
        engine.implied_vol(1.0, 42, 40, 0.5, 0.1, "call")      # below intrinsic-ish bound
    with pytest.raises(engine.EngineError):
        engine.implied_vol(50.0, 42, 40, 0.5, 0.1, "call")     # above S


def test_binomial_converges_to_black_scholes():
    """Hull 13.10 / 21.1: CRR European converges to BSM. Within 1 cent at 500 steps."""
    bs = engine.bs_price(42, 40, 0.5, 0.1, 0.2, "call")
    assert engine.binomial_price(42, 40, 0.5, 0.1, 0.2, "call", steps=500) == pytest.approx(bs, abs=0.01)


def test_american_put_is_worth_at_least_european():
    eu = engine.binomial_price(40, 45, 1.0, 0.08, 0.3, "put", steps=300, american=False)
    am = engine.binomial_price(40, 45, 1.0, 0.08, 0.3, "put", steps=300, american=True)
    assert am >= eu - 1e-9
    assert am > eu + 0.01   # deep enough in the money with r > 0 that early exercise has value


def test_american_call_no_dividends_equals_european():
    """Hull 11.5: never optimal to exercise an American call early on a non-dividend stock."""
    eu = engine.binomial_price(42, 40, 0.5, 0.1, 0.2, "call", steps=300, american=False)
    am = engine.binomial_price(42, 40, 0.5, 0.1, 0.2, "call", steps=300, american=True)
    assert am == pytest.approx(eu, abs=1e-6)


# ----------------------------------------------------------------------------- refusals

def test_engine_refuses_bad_inputs():
    with pytest.raises(engine.EngineError):
        engine.bs_price(42, 40, 0.5, 0.1, 0.0, "call")       # no vol, no price
    with pytest.raises(engine.EngineError):
        engine.bs_price(42, 40, -0.1, 0.1, 0.2, "call")
    with pytest.raises(engine.EngineError):
        engine.bs_price(42, 40, 0.5, 0.1, 0.2, "straddle")   # not a thing the engine prices
