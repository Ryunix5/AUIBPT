# Schedule builder functionality for AUIBPT

import io
import csv
import json
import re
from typing import List, Dict, Set, Tuple

from ..utils.constants import LA_REQUIREMENTS, LA_CATEGORY, LA_QUANT_BOTH, MAJOR_MAP, DEGREE_TOTAL, DIFFICULTY_WEIGHT_MAP
from ..utils.helpers import (
    la_completed_counts, la_remaining, la_recommend_pool, 
    _eligible_major_rows, _parse_prereq_codes, _credits_from_str,
    _credits_completed
)

def get_semester_weights(difficulty: str) -> tuple[float, float]:
    """Get semester weights based on difficulty."""
    return DIFFICULTY_WEIGHT_MAP.get(difficulty, DIFFICULTY_WEIGHT_MAP["Medium"])

def build_semester_schedule(
    major_key: str,
    target_credits: int,
    taken_codes: Set[str],
    rows_all: List[Dict],
    difficulty: str
) -> Tuple[List[Dict], Dict[str,int], Dict[str,int], int]:
    """Build a semester schedule."""
    used_codes: Set[str] = set(c.upper() for c in taken_codes)
    code_to_row = {r["code"].upper(): r for r in rows_all}

    la_counts = la_completed_counts(used_codes)
    la_remain = la_remaining(la_counts, used_codes)
    la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)

    la_flat: List[Tuple[str, Dict]] = []
    for cat, lst in la_pool_by_cat.items():
        for r in lst:
            la_flat.append((cat, r))

    major_info = MAJOR_MAP[major_key]
    major_pool = _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])

    def cr_of(r: Dict) -> int:
        return _credits_from_str(r.get("credits"))

    schedule: List[Dict] = []
    cur_credits = 0

    def try_add_row(r: Dict, origin: str) -> bool:
        nonlocal cur_credits
        cu = r["code"].upper()
        if cu in used_codes:
            return False
        reqs = _parse_prereq_codes(r.get("prereqs",""))
        if any(rc not in used_codes for rc in reqs):
            return False
        c = cr_of(r)
        if cur_credits + c > target_credits:
            return False
        schedule.append(r)
        used_codes.add(cu)
        cur_credits += c
        return True

    # Add required quantitative courses first
    for q_code in ["CSC101", "MAT101"]:
        if la_remain.get("Quantitative", 0) > 0 and q_code not in used_codes:
            rq = code_to_row.get(q_code)
            if rq and try_add_row(rq, origin="LA:Quantitative"):
                la_counts = la_completed_counts(used_codes)
                la_remain = la_remaining(la_counts, used_codes)
                la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
                la_flat = []
                for cat, lst in la_pool_by_cat.items():
                    for r in lst:
                        la_flat.append((cat, r))

    major_weight, la_weight = get_semester_weights(difficulty)
    major_budget = major_weight
    la_budget = la_weight

    guard = 0
    MAX_ITERS = 1000

    while cur_credits < target_credits and guard < MAX_ITERS:
        guard += 1

        major_pool = [r for r in _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])
                      if r["code"].upper() not in used_codes]

        la_counts = la_completed_counts(used_codes)
        la_remain = la_remaining(la_counts, used_codes)
        la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
        la_flat = [(cat, r) for cat, lst in la_pool_by_cat.items() for r in lst
                   if r["code"].upper() not in used_codes]

        la_any_needed = any(v > 0 for v in la_remain.values())
        picked = False

        def pick_major_first() -> bool:
            if not la_any_needed:
                return True
            if la_budget > 0 and la_flat:
                return False
            return True

        try_major = pick_major_first()

        for bucket in (["major", "la"] if try_major else ["la", "major"]):
            if bucket == "major":
                if major_budget <= 0 or not major_pool:
                    continue
                for r in major_pool:
                    if try_add_row(r, origin="Major"):
                        major_budget -= 1
                        picked = True
                        break
                if picked:
                    break
            else:
                if la_budget <= 0 or not la_flat:
                    continue
                for (cat, r) in la_flat:
                    if la_remain.get(cat, 0) <= 0:
                        continue
                    if try_add_row(r, origin=f"LA:{cat}"):
                        la_budget -= 1
                        picked = True
                        break
                if picked:
                    break

        if not picked:
            break

        if major_budget <= 0 and la_budget <= 0:
            major_budget = major_weight
            la_budget = la_weight

    return schedule, la_counts, la_remain, cur_credits

def _rebuild_pools(major_key: str, taken_codes: Set[str], rows_all: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """Rebuild LA and major pools."""
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, rows_all, la_remain)
    major_pool = _eligible_major_rows(taken_codes, rows_all, MAJOR_MAP[major_key]["prefixes"])
    return la_pool, major_pool

def _export_schedule_csv(slots: List[Dict]) -> bytes:
    """Export schedule as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["code", "title", "credits", "category", "prereqs"])
    for s in slots:
        r = s["candidates"][s["current_idx"]]
        writer.writerow([r["code"], r["title"], r.get("credits") or "", s["origin"], r.get("prereqs") or ""])
    return output.getvalue().encode("utf-8")

def export_schedule_json(slots):
    """Export schedule as JSON."""
    data = [{
        "origin": s["origin"],
        "current_idx": s["current_idx"],
        "candidates": [{
            "code": r["code"], "title": r["title"],
            "credits": r.get("credits"), "prereqs": r.get("prereqs")
        } for r in s["candidates"]]
    } for s in slots]
    return json.dumps({"version":"1.0","slots":data}, ensure_ascii=False, indent=2).encode("utf-8")

def import_schedule_json(payload_bytes):
    """Import schedule from JSON."""
    obj = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(obj, dict) or "slots" not in obj:
        raise ValueError("Invalid schedule file.")
    
    new_slots = []
    for s in obj["slots"]:
        if not {"origin","current_idx","candidates"} <= set(s):
            continue
        cand_rows = []
        for r in s["candidates"]:
            if "code" in r and "title" in r:
                cand_rows.append({"code":r["code"],"title":r["title"],"credits":r.get("credits"),"prereqs":r.get("prereqs")})
        if cand_rows:
            new_slots.append({
                "id": f"import-{len(new_slots)}",
                "origin": s["origin"],
                "candidates": cand_rows,
                "current_idx": int(s["current_idx"]),
                "locked": bool(s.get("locked", False)),
            })
    return new_slots

def _auto_top_up(major_key: str, target_credits: int, taken_codes: Set[str], slots: List[Dict], rows_all: List[Dict]) -> None:
    """Auto top-up schedule with additional courses."""
    current_total = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in slots)
    if current_total >= target_credits:
        return
    
    used = set(taken_codes) | {s["candidates"][s["current_idx"]]["code"].upper() for s in slots}
    la_pool, major_pool = _rebuild_pools(major_key, used, rows_all)

    la_candidates = [r for pool in la_pool.values() for r in pool if r["code"].upper() not in used]
    major_candidates = [r for r in major_pool if r["code"].upper() not in used]
    
    for origin, cand_list in [("LA:Any", la_candidates), ("Major", major_candidates)]:
        for r in cand_list:
            cr = _credits_from_str(r.get("credits"))
            if current_total + cr <= target_credits:
                slots.append({
                    "id": f"extra-{origin}-{len(slots)}",
                    "origin": origin,
                    "candidates": [r],
                    "current_idx": 0,
                    "locked": False,
                })
                used.add(r["code"].upper())
                current_total += cr
                if current_total >= target_credits:
                    return

def _toggle_lock(slot):
    """Toggle lock status of a slot."""
    if "locked" not in slot:
        slot["locked"] = True
    else:
        slot["locked"] = not slot["locked"]

def _push_swap(slot_idx, prev_idx, swap_history):
    """Push swap to history."""
    swap_history.append((slot_idx, prev_idx))

def _undo_swap(swap_history, schedule_slots):
    """Undo last swap."""
    if not swap_history:
        return False
    slot_idx, prev_idx = swap_history.pop()
    if 0 <= slot_idx < len(schedule_slots):
        schedule_slots[slot_idx]["current_idx"] = prev_idx
        return True
    return False
