"""
Kill-criteria reporting for Stage 1 and Stage 2.

These checks are automatic and never softened. A clean negative result is
an acceptable and valuable outcome -- the code does not bias toward positive.
"""

import numpy as np


SEP = "=" * 64


def report_stage1(all_results: dict) -> None:
    """
    Stage 1 kill criteria (CLAUDE.md S6):
      - Both grokfast_full AND grokfast_emb > 1% worse than baseline -> DEAD
      - Also reports wall-clock overhead and arm-4 momentum isolation check
    """
    print(f"\n{SEP}")
    print("KILL-CRITERIA REPORT  (Stage 1)")
    print(SEP)

    if "baseline" not in all_results or not all_results["baseline"].get("val_loss"):
        print("ERROR: baseline results missing -- cannot evaluate.")
        return

    base_final = all_results["baseline"]["val_loss"][-1]
    print(f"baseline final val_loss: {base_final:.6f}\n")

    dead_flags: list[bool] = []
    for arm in ("grokfast_full", "grokfast_emb"):
        if arm not in all_results or not all_results[arm].get("val_loss"):
            continue
        arm_final = all_results[arm]["val_loss"][-1]
        pct       = (arm_final - base_final) / base_final * 100.0
        dead      = arm_final > base_final * 1.01
        dead_flags.append(dead)
        tag = "DEAD  (>1% worse)" if dead else "ALIVE"
        print(f"  {arm:<28}  final={arm_final:.6f}  {pct:+.2f}%  -> {tag}")

    print()
    if len(dead_flags) == 2 and all(dead_flags):
        print(">>> VERDICT: BOTH grokfast arms >1% worse. DEAD. Do not proceed to Stage 2.")
    elif any(dead_flags):
        print(">>> VERDICT: Mixed result. Review curves before deciding on Stage 2.")
    else:
        print(">>> VERDICT: Both Grokfast arms beat baseline. ALIVE. Proceed to Stage 2.")

    # Wall-clock overhead
    print()
    base_times = all_results["baseline"].get("step_times_ms", [])
    if base_times:
        base_avg = sum(base_times) / len(base_times)
        print(f"Wall-clock overhead vs baseline (avg {base_avg:.1f}ms/step):")
        for arm in ("grokfast_full", "grokfast_emb", "grokfast_nomomentum"):
            if arm not in all_results or not all_results[arm].get("step_times_ms"):
                continue
            t = all_results[arm]["step_times_ms"]
            avg = sum(t) / len(t)
            pct = (avg - base_avg) / max(1.0, base_avg) * 100.0
            flag = "  *** >5% COST CONCERN ***" if pct > 5.0 else ""
            print(f"  {arm:<28}  {avg:.1f}ms  ({pct:+.1f}%){flag}")

    # Arm-4 momentum isolation
    print()
    if "grokfast_nomomentum" in all_results and "grokfast_full" in all_results:
        nm   = all_results["grokfast_nomomentum"]["val_loss"][-1]
        gf   = all_results["grokfast_full"]["val_loss"][-1]
        diff = nm - gf
        print(f"Arm-4 momentum isolation: nomomentum={nm:.6f}  full={gf:.6f}  diff={diff:+.6f}")
        if abs(diff) < 0.005:
            print("  -> Similar: Grokfast benefit may be explained by momentum alone.")
        elif diff > 0:
            print("  -> nomomentum worse: standard momentum + Grokfast is synergistic.")
        else:
            print("  -> nomomentum better: unexpected -- investigate.")

    print(SEP)


def report_stage2(all_results: dict, full_verdict: bool = False) -> None:
    """
    Stage 2 kill criteria:
      - grokfast_emb must beat baseline on 2/2 seeds -> ALIVE
      - 1/2 or 0/2 -> DEAD

    full_verdict=False: per-GPU partial report (one seed). Does not print DEAD
        prematurely -- waits for --report-only to confirm with both seeds.
    full_verdict=True: definitive verdict (called by --report-only).
    """
    print(f"\n{SEP}")
    print("KILL-CRITERIA REPORT  (Stage 2)")
    print(SEP)

    seeds = sorted({r["seed"] for r in all_results.values() if "seed" in r})
    wins  = 0
    rows  = []

    for seed in seeds:
        base_key = f"baseline_seed{seed}"
        emb_key  = f"grokfast_emb_seed{seed}"
        if base_key not in all_results or emb_key not in all_results:
            print(f"  seed {seed}: MISSING results -- skipping")
            continue
        base_v = all_results[base_key]["val_loss"][-1]
        emb_v  = all_results[emb_key]["val_loss"][-1]
        pct    = (emb_v - base_v) / base_v * 100.0
        win    = emb_v < base_v
        wins  += int(win)
        rows.append((seed, base_v, emb_v, pct, "WIN" if win else "LOSS"))
        print(f"  seed {seed}:  baseline={base_v:.6f}  emb={emb_v:.6f}  {pct:+.2f}%  -> {'WIN' if win else 'LOSS'}")

    if rows:
        base_mean = np.mean([r[1] for r in rows])
        base_std  = np.std( [r[1] for r in rows])
        emb_mean  = np.mean([r[2] for r in rows])
        emb_std   = np.std( [r[2] for r in rows])
        print(f"\n  baseline  mean={base_mean:.6f}  std={base_std:.6f}")
        print(f"  emb       mean={emb_mean:.6f}  std={emb_std:.6f}")

    n_seeds = len(rows)
    print()
    if n_seeds == 0:
        print(">>> VERDICT: no complete seed pairs -- cannot evaluate.")
    elif not full_verdict and n_seeds < 2:
        print(f">>> PARTIAL ({n_seeds}/2 seeds): grokfast_emb wins {wins}/{n_seeds} so far.")
        print(">>> Run --report-only after both GPUs finish for final verdict.")
    elif wins >= 2:
        print(f">>> VERDICT: grokfast_emb wins on {wins}/{n_seeds} seeds. ALIVE.")
        print(">>> Proceed to HuggingFace release and write-up.")
    else:
        print(f">>> VERDICT: grokfast_emb wins on only {wins}/{n_seeds} seeds. DEAD.")
        print(">>> Do not release. Re-examine Stage 1 result for confounds.")

    print(SEP)


def report_kill_criteria(
    all_results: dict,
    stage: int,
    full_verdict: bool = False,
) -> None:
    """Dispatch to the correct stage report."""
    if stage == 1:
        report_stage1(all_results)
    elif stage == 2:
        report_stage2(all_results, full_verdict=full_verdict)
    else:
        raise ValueError(f"Unknown stage: {stage}")
