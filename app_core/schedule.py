"""Schedule builder logic and UI."""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

from .knowledge import DEGREE_TOTAL, MAJOR_MAP, get_current_major_key
from .ui_controls import ui_int, ui_multi, ui_select

LA_REQUIREMENTS = {
    "General": 1,
    "Communication": 3,
    "Quantitative": 2,
    "Humanities": 4,
    "SocialScience": 2,
    "NaturalScience": 2,
}

LA_CATEGORY = {
    "UNI101": "General",
    "ENL101": "Communication",
    "ENL201": "Communication",
    "ENL210": "Communication",
    "CSC101": "Quantitative",
    "MAT101": "Quantitative",
    "HIS101": "Humanities",
    "HIS102": "Humanities",
    "HIS105": "Humanities",
    "HUM101": "Humanities",
    "LIT101": "Humanities",
    "PHA210": "Humanities",
    "PHI101": "Humanities",
    "POL125": "Humanities",
    "TLD100": "Humanities",
    "TLD101": "Humanities",
    "TLD102": "Humanities",
    "TLD103": "Humanities",
    "COM101": "SocialScience",
    "ECO101": "SocialScience",
    "FIN101": "SocialScience",
    "HCT108": "SocialScience",
    "MIS101": "SocialScience",
    "POL101": "SocialScience",
    "POL112": "SocialScience",
    "POL191": "SocialScience",
    "PSY101": "SocialScience",
    "SOC101": "SocialScience",
    "CHE100": "NaturalScience",
    "ENV201": "NaturalScience",
    "GEO101": "NaturalScience",
    "PHY100": "NaturalScience",
    "PHY105": "NaturalScience",
}

LA_QUANT_BOTH = {"CSC101", "MAT101"}
MAJOR_WEIGHT, LA_WEIGHT = 3, 1
DIFFICULTY_WEIGHT_MAP = {"Easy": (2.0, 1.0), "Medium": (3.0, 1.0), "Hard": (4.0, 1.0)}


def _parse_prereq_codes(prereq_text: str) -> List[str]:
    if not prereq_text:
        return []
    codes: List[str] = []
    for part in re.split(r"[;,/]+", prereq_text):
        for match in re.finditer(r"\b([A-Za-z]{2,4})\s*-?\s*(\d{3})\b", part):
            codes.append((match.group(1) + match.group(2)).upper())
    return sorted(set(codes))


def _credits_from_str(value: str) -> int:
    if value is None:
        return 3
    string_value = str(value).strip()
    if not string_value:
        return 3
    match = re.search(r"\d+", string_value)
    return int(match.group(0)) if match else 3


def la_completed_counts(taken_codes: Set[str]) -> Dict[str, int]:
    counts = {key: 0 for key in LA_REQUIREMENTS}
    for code in taken_codes:
        category = LA_CATEGORY.get(code.upper())
        if category:
            counts[category] += 1
    return counts


def la_remaining(counts: Dict[str, int], taken_codes: Set[str]) -> Dict[str, int]:
    remain: Dict[str, int] = {}
    for category, need in LA_REQUIREMENTS.items():
        have = counts.get(category, 0)
        remain[category] = max(0, need - have)
    have_both = LA_QUANT_BOTH.issubset({code.upper() for code in taken_codes})
    remain["Quantitative"] = 0 if have_both else len(LA_QUANT_BOTH - {code.upper() for code in taken_codes})
    return remain


def la_recommend_pool(taken_codes: Set[str], rows_scope: List[Dict], remain: Dict[str, int]) -> Dict[str, List[Dict]]:
    taken_upper = {code.upper() for code in taken_codes}
    code_to_row = {row["code"].upper(): row for row in rows_scope}
    by_cat = {key: [] for key in LA_REQUIREMENTS}
    for code, category in LA_CATEGORY.items():
        row = code_to_row.get(code)
        if not row or remain.get(category, 0) <= 0 or code in taken_upper:
            continue
        req_codes = _parse_prereq_codes(row.get("prereqs", ""))
        if not all(req in taken_upper for req in req_codes):
            continue
        by_cat[category].append(row)
    for category, pool in by_cat.items():
        pool.sort(
            key=lambda row: (
                int(re.search(r"(\d{3})$", row["code"]).group(1)) if re.search(r"(\d{3})$", row["code"]) else 999,
                row.get("title", ""),
            )
        )
    return by_cat


def _eligible_major_rows(taken_codes: Set[str], rows_scope: List[Dict], prefixes: Tuple[str, ...]) -> List[Dict]:
    taken_upper = {code.upper() for code in taken_codes}
    pool = []
    for row in rows_scope:
        code = row["code"].upper()
        if code in taken_upper or not code.startswith(prefixes):
            continue
        req_codes = _parse_prereq_codes(row.get("prereqs", ""))
        if req_codes and not all(req in taken_upper for req in req_codes):
            continue
        pool.append(row)

    def score(row):
        req = _parse_prereq_codes(row.get("prereqs", ""))
        level_match = re.search(r"(\d{3})$", row["code"].upper())
        level = int(level_match.group(1)) if level_match else 0
        return (-len(req), level, row.get("title", ""))

    return sorted(pool, key=score)


def _credits_completed(taken_codes: Set[str], rows_all: List[Dict]) -> int:
    index = {row["code"].upper(): row for row in rows_all}
    total = 0
    for code in taken_codes:
        row = index.get(code.upper())
        if row:
            total += _credits_from_str(row.get("credits"))
    return total


def student_context_from_taken(rows_all: List[Dict], taken_codes: Set[str]) -> str:
    if not taken_codes:
        return "Completed: (none)"
    index = {row["code"].upper(): row for row in rows_all}
    items: List[str] = []
    for code in sorted({value.upper() for value in taken_codes}):
        row = index.get(code)
        if not row:
            continue
        credits = row.get("credits") or ""
        title = row.get("title") or ""
        items.append(f"{code} ({title}; {credits} cr)")
    completed_credits = _credits_completed(taken_codes, rows_all)
    return "Completed (" + str(completed_credits) + " credits): " + "; ".join(items)


def get_semester_weights(difficulty: str) -> Tuple[float, float]:
    return DIFFICULTY_WEIGHT_MAP.get(difficulty, DIFFICULTY_WEIGHT_MAP["Medium"])


def build_semester_schedule(
    *,
    major_key: str,
    target_credits: int,
    taken_codes: Set[str],
    rows_all: List[Dict],
    difficulty: str,
    desired_major: Optional[int] = None,
    desired_la: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, int], Dict[str, int], int]:
    used_codes: Set[str] = {code.upper() for code in taken_codes}
    code_to_row = {row["code"].upper(): row for row in rows_all}

    la_counts = la_completed_counts(used_codes)
    la_remain = la_remaining(la_counts, used_codes)
    la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)

    la_flat: List[Tuple[str, Dict]] = []
    for category, rows in la_pool_by_cat.items():
        for row in rows:
            la_flat.append((category, row))

    major_info = MAJOR_MAP[major_key]
    major_pool = _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])

    def credits_of(row: Dict) -> int:
        return _credits_from_str(row.get("credits"))

    schedule: List[Dict] = []
    current_credits = 0

    def try_add_row(row: Dict) -> bool:
        nonlocal current_credits
        code_upper = row["code"].upper()
        if code_upper in used_codes:
            return False
        requirements = _parse_prereq_codes(row.get("prereqs", ""))
        if any(req not in used_codes for req in requirements):
            return False
        credits = credits_of(row)
        if current_credits + credits > target_credits:
            return False
        schedule.append(row)
        used_codes.add(code_upper)
        current_credits += credits
        return True

    for q_code in ["CSC101", "MAT101"]:
        if la_remain.get("Quantitative", 0) > 0 and q_code not in used_codes:
            row = code_to_row.get(q_code)
            if row and try_add_row(row):
                la_counts = la_completed_counts(used_codes)
                la_remain = la_remaining(la_counts, used_codes)
                la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
                la_flat = []
                for category, rows in la_pool_by_cat.items():
                    for row_item in rows:
                        la_flat.append((category, row_item))

    if desired_major is not None or desired_la is not None:
        avg_cr = 3
        est_slots = max(1, min(7, target_credits // avg_cr))
        if desired_major is None and desired_la is not None:
            desired_major = max(0, est_slots - int(desired_la))
        if desired_la is None and desired_major is not None:
            desired_la = max(0, est_slots - int(desired_major))
        desired_major = max(0, int(desired_major or 0))
        desired_la = max(0, int(desired_la or 0))
    else:
        major_weight, la_weight = get_semester_weights(difficulty)
        avg_cr = 3
        total_slots = max(1, min(7, target_credits // avg_cr))
        if major_weight + la_weight <= 0:
            major_goal, la_goal = total_slots, 0
        else:
            major_goal = round(total_slots * (major_weight / (major_weight + la_weight)))
            la_goal = total_slots - major_goal
        desired_major, desired_la = max(0, major_goal), max(0, la_goal)

    major_remaining = int(desired_major)
    la_remaining_goal = int(desired_la)

    guard, max_iters = 0, 1000
    while current_credits < target_credits and guard < max_iters:
        guard += 1
        major_pool = [
            row
            for row in _eligible_major_rows(used_codes, rows_all, major_info["prefixes"])
            if row["code"].upper() not in used_codes
        ]

        la_counts = la_completed_counts(used_codes)
        la_remain = la_remaining(la_counts, used_codes)
        la_pool_by_cat = la_recommend_pool(used_codes, rows_all, la_remain)
        la_flat = [
            (category, row)
            for category, rows in la_pool_by_cat.items()
            for row in rows
            if row["code"].upper() not in used_codes
        ]
        la_any_available = bool(la_flat)
        picked = False

        def pick_major_first() -> bool:
            if major_remaining > 0 and la_remaining_goal > 0:
                if any(value > 0 for value in la_remain.values()):
                    return False
                return True
            if major_remaining > 0:
                return True
            if la_remaining_goal > 0:
                return False
            return bool(major_pool)

        try_major = pick_major_first()

        for bucket in (["major", "la"] if try_major else ["la", "major"]):
            if bucket == "major":
                if major_remaining <= 0 or not major_pool:
                    continue
                for row in major_pool:
                    if try_add_row(row):
                        major_remaining -= 1
                        picked = True
                        break
                if picked:
                    break
            else:
                if la_remaining_goal <= 0 or not la_any_available:
                    continue
                for category, row in la_flat:
                    if la_remain.get(category, 0) <= 0 and any(value > 0 for value in la_remain.values()):
                        continue
                    if try_add_row(row):
                        la_remaining_goal -= 1
                        picked = True
                        break
                if picked:
                    break

        if not picked:
            for row in major_pool:
                if try_add_row(row):
                    picked = True
                    break
            if not picked:
                for _, row in la_flat:
                    if try_add_row(row):
                        picked = True
                        break
        if not picked:
            break
        if major_remaining <= 0 and la_remaining_goal <= 0 and current_credits >= target_credits - 2:
            break

    return schedule, la_counts, la_remain, current_credits


def _rebuild_pools(major_key: str, taken_codes: Set[str], rows_all: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    la_counts = la_completed_counts(taken_codes)
    la_remain = la_remaining(la_counts, taken_codes)
    la_pool = la_recommend_pool(taken_codes, rows_all, la_remain)
    major_pool = _eligible_major_rows(taken_codes, rows_all, MAJOR_MAP[major_key]["prefixes"])
    return la_pool, major_pool


def _export_schedule_csv(slots: List[Dict]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["code", "title", "credits", "category", "prereqs"])
    for slot in slots:
        row = slot["candidates"][slot["current_idx"]]
        writer.writerow([
            row["code"],
            row["title"],
            row.get("credits") or "",
            slot["origin"],
            row.get("prereqs") or "",
        ])
    return output.getvalue().encode("utf-8")


def export_schedule_json(slots: List[Dict]) -> bytes:
    data = [
        {
            "origin": slot["origin"],
            "current_idx": slot["current_idx"],
            "candidates": [
                {
                    "code": row["code"],
                    "title": row["title"],
                    "credits": row.get("credits"),
                    "prereqs": row.get("prereqs"),
                }
                for row in slot["candidates"]
            ],
        }
        for slot in slots
    ]
    return json.dumps({"version": "1.0", "slots": data}, ensure_ascii=False, indent=2).encode("utf-8")


def import_schedule_json(payload_bytes: bytes):
    obj = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(obj, dict) or "slots" not in obj:
        raise ValueError("Invalid schedule file.")
    new_slots = []
    for slot in obj["slots"]:
        if not {"origin", "current_idx", "candidates"} <= set(slot):
            continue
        candidates = []
        for row in slot["candidates"]:
            if "code" in row and "title" in row:
                candidates.append(
                    {
                        "code": row["code"],
                        "title": row["title"],
                        "credits": row.get("credits"),
                        "prereqs": row.get("prereqs"),
                    }
                )
        if candidates:
            new_slots.append(
                {
                    "id": f"import-{len(new_slots)}",
                    "origin": slot["origin"],
                    "candidates": candidates,
                    "current_idx": int(slot["current_idx"]),
                    "locked": bool(slot.get("locked", False)),
                }
            )
    return new_slots


def _auto_top_up(major_key: str, target_credits: int, taken_codes: Set[str], slots: List[Dict], rows_all: List[Dict]) -> None:
    def total() -> int:
        return sum(_credits_from_str(slot["candidates"][slot["current_idx"]].get("credits")) for slot in slots)

    current_total = total()
    if current_total >= target_credits:
        return
    used = set(taken_codes) | {slot["candidates"][slot["current_idx"]]["code"].upper() for slot in slots}
    la_pool, major_pool = _rebuild_pools(major_key, used, rows_all)
    la_candidates = [row for pool in la_pool.values() for row in pool if row["code"].upper() not in used]
    major_candidates = [row for row in major_pool if row["code"].upper() not in used]
    for origin, candidate_list in [("LA:Any", la_candidates), ("Major", major_candidates)]:
        for row in candidate_list:
            credits = _credits_from_str(row.get("credits"))
            if current_total + credits <= target_credits:
                slots.append(
                    {
                        "id": f"extra-{origin}-{len(slots)}",
                        "origin": origin,
                        "candidates": [row],
                        "current_idx": 0,
                        "locked": False,
                    }
                )
                used.add(row["code"].upper())
                current_total += credits
                if current_total >= target_credits:
                    return


def _toggle_lock(slot: Dict) -> None:
    slot["locked"] = not bool(slot.get("locked"))


def _push_swap(slot_idx: int, prev_idx: int) -> None:
    st.session_state.swap_history.append((slot_idx, prev_idx))


def _undo_swap() -> bool:
    if not st.session_state.swap_history:
        return False
    slot_idx, prev_idx = st.session_state.swap_history.pop()
    if 0 <= slot_idx < len(st.session_state.schedule_slots):
        st.session_state.schedule_slots[slot_idx]["current_idx"] = prev_idx
        return True
    return False


def render_schedule_builder(rows_all, vs, bm25):
    st.markdown("### Schedule Builder")

    mcol1, mcol2 = st.columns([1, 1])
    with mcol2:
        st.session_state.mobile_mode = st.toggle(
            "Mobile-friendly controls",
            value=st.session_state.get("mobile_mode", False),
            help="Use touch controls on phones (no typing).",
        )

    if "schedule_difficulty" not in st.session_state:
        st.session_state.schedule_difficulty = "Medium"
    if "schedule_custom_mode" not in st.session_state:
        st.session_state.schedule_custom_mode = False
    if "schedule_target_credits" not in st.session_state:
        st.session_state.schedule_target_credits = 15
    st.session_state.setdefault("schedule_major_count", 3)
    st.session_state.setdefault("schedule_la_count", 2)

    diff_opts = ["Easy", "Medium", "Hard", "Custom"]
    current_choice = "Custom" if st.session_state.schedule_custom_mode else st.session_state.schedule_difficulty
    difficulty_choice = st.radio(
        "Semester difficulty",
        options=diff_opts,
        index=diff_opts.index(current_choice),
        horizontal=True,
        help="Easy favors LA; Hard favors Major; Custom lets you set exact counts.",
    )
    if difficulty_choice == "Custom":
        st.session_state.schedule_custom_mode = True
    else:
        st.session_state.schedule_custom_mode = False
        st.session_state.schedule_difficulty = difficulty_choice

    options = sorted(MAJOR_MAP.keys())
    current = get_current_major_key()
    major_key = ui_select(
        "Major / program",
        options,
        default=(current if current in options else (options[0] if options else None)),
        key="schedule_major_key",
    )
    _major_key = get_current_major_key()

    target_credits = st.slider(
        "Target credits",
        min_value=9,
        max_value=21,
        value=int(st.session_state.schedule_target_credits),
        step=1,
    )
    st.session_state.schedule_target_credits = target_credits

    if st.session_state.schedule_custom_mode:
        cols_counts = st.columns(2)
        with cols_counts[0]:
            st.session_state.schedule_major_count = ui_int(
                "Major courses this term",
                min_value=0,
                max_value=7,
                step=1,
                value=int(st.session_state.get("schedule_major_count", 3)),
                key="ui_major_count",
            )
        with cols_counts[1]:
            st.session_state.schedule_la_count = ui_int(
                "Liberal Arts courses this term",
                min_value=0,
                max_value=7,
                step=1,
                value=int(st.session_state.get("schedule_la_count", 2)),
                key="ui_la_count",
            )

    picker_scope = st.radio("Completed-course picker scope:", ["Major only", "Liberal Arts only", "Both"], horizontal=True)
    major_prefixes = MAJOR_MAP[_major_key]["prefixes"]
    major_only_rows = [row for row in rows_all if row["code"].upper().startswith(major_prefixes)]
    la_only_rows = [row for row in rows_all if row["code"].upper() in LA_CATEGORY]
    if picker_scope == "Major only":
        picker_rows = major_only_rows
    elif picker_scope == "Liberal Arts only":
        picker_rows = la_only_rows
    else:
        seen = set()
        picker_rows = []
        for row in major_only_rows + la_only_rows:
            code_upper = row["code"].upper()
            if code_upper not in seen:
                picker_rows.append(row)
                seen.add(code_upper)

    labels = [f"{row['code']} — {row['title']}" for row in picker_rows]
    label_to_code = {f"{row['code']} — {row['title']}": row["code"].upper() for row in picker_rows}
    visible_codes = set(label_to_code.values())

    preselected_labels = [label for label, code in label_to_code.items() if code in st.session_state.completed_codes_all]
    picked_labels = ui_multi(
        "I have completed:",
        labels,
        default=preselected_labels,
        key="completed_picker",
    )
    picked_visible_codes = {label_to_code[label] for label in picked_labels}
    hidden_kept = st.session_state.completed_codes_all - visible_codes
    st.session_state.completed_codes_all = hidden_kept | picked_visible_codes
    taken_codes_all = set(st.session_state.completed_codes_all)

    completed_credits = _credits_completed(taken_codes_all, rows_all)
    degree_total = DEGREE_TOTAL.get(_major_key, 126)
    st.caption(f"Progress: {completed_credits} / {degree_total} credits • Target this term: {target_credits}")

    col_build_a, col_build_b, col_build_c, col_build_d, col_build_e = st.columns([0.35, 0.2, 0.2, 0.15, 0.1])
    with col_build_a:
        build_btn = st.button("Build schedule", use_container_width=True)
    with col_build_b:
        reset_btn = st.button("Reset", use_container_width=True)
    with col_build_c:
        topup_btn = st.button(
            "Auto top-up",
            use_container_width=True,
            disabled=not st.session_state.get("schedule_slots"),
        )
    with col_build_d:
        undo_btn = st.button(
            "Undo swap",
            use_container_width=True,
            disabled=not st.session_state.get("swap_history"),
        )
    with col_build_e:
        if st.button("Close", use_container_width=True):
            st.session_state.show_schedule = False
            st.rerun()

    if reset_btn:
        st.session_state.schedule_slots = []
        st.session_state.schedule_planned_credits = 0
        st.rerun()

    if build_btn:
        desired_major = st.session_state.schedule_major_count if st.session_state.schedule_custom_mode else None
        desired_la = st.session_state.schedule_la_count if st.session_state.schedule_custom_mode else None
        schedule, la_counts, la_remain, planned_credits = build_semester_schedule(
            major_key=_major_key,
            target_credits=target_credits,
            taken_codes=taken_codes_all,
            rows_all=rows_all,
            difficulty=st.session_state.get("schedule_difficulty", "Medium"),
            desired_major=desired_major,
            desired_la=desired_la,
        )
        la_pool, major_pool = _rebuild_pools(_major_key, taken_codes_all, rows_all)
        slots = []
        used = set(taken_codes_all)
        for idx, course in enumerate(schedule):
            if course["code"].upper() in LA_CATEGORY:
                origin = f"LA:{LA_CATEGORY[course['code'].upper()]}"
                pool = la_pool.get(LA_CATEGORY[course["code"].upper()], [])
            else:
                origin = "Major"
                pool = major_pool
            candidates, seen_codes = [], set()
            candidates.append(course)
            seen_codes.add(course["code"].upper())
            for row in pool:
                code_upper = row["code"].upper()
                if code_upper not in seen_codes and code_upper not in used:
                    candidates.append(row)
                    seen_codes.add(code_upper)
            slots.append(
                {
                    "id": f"{origin}-{idx}",
                    "origin": origin,
                    "candidates": candidates,
                    "current_idx": 0,
                    "locked": False,
                }
            )
            used.add(course["code"].upper())
        st.session_state.schedule_slots = slots
        st.session_state.schedule_planned_credits = sum(
            _credits_from_str(slot["candidates"][0].get("credits")) for slot in slots
        )

    if topup_btn and st.session_state.get("schedule_slots"):
        _auto_top_up(_major_key, target_credits, taken_codes_all, st.session_state.schedule_slots, rows_all)
        st.session_state.schedule_planned_credits = sum(
            _credits_from_str(slot["candidates"][slot["current_idx"]].get("credits"))
            for slot in st.session_state.schedule_slots
        )
        st.rerun()

    if undo_btn:
        if _undo_swap():
            st.rerun()
        else:
            st.info("Nothing to undo.")

    if st.session_state.get("schedule_slots"):
        st.markdown("### Suggested schedule")
        new_total = 0
        for idx, slot in enumerate(st.session_state.schedule_slots):
            current = slot["candidates"][slot["current_idx"]]
            prereqs = current.get("prereqs") or "None/Unknown"
            credits = current.get("credits") or "Unknown"
            cols = st.columns([0.64, 0.12, 0.12, 0.12])
            with cols[0]:
                st.markdown(
                    f"**{current['code']} — {current['title']}**  \n"
                    f"Category: {slot['origin']} • Credits: {credits} • Prereqs: {prereqs}\n"
                    f"Status: {'Locked' if slot.get('locked') else 'Unlocked'}"
                )
                why_bits = []
                if current["code"].upper() in LA_CATEGORY:
                    why_bits.append(f"meets {LA_CATEGORY[current['code'].upper()]}")
                else:
                    why_bits.append("major requirement/elective")
                reqs = _parse_prereq_codes(current.get("prereqs", ""))
                if not reqs:
                    why_bits.append("no explicit prerequisites")
                else:
                    if all(req in taken_codes_all for req in reqs):
                        why_bits.append("prerequisites satisfied")
                    else:
                        why_bits.append("prerequisites satisfied during planning")
                st.caption("Why this? " + " • ".join(why_bits))
            with cols[1]:
                if st.button("Swap", key=f"swap_{slot['id']}", disabled=bool(slot.get("locked", False))):
                    current_used = {
                        slot_info["candidates"][slot_info["current_idx"]]["code"].upper()
                        for slot_info in st.session_state.schedule_slots
                    }
                    current_used.discard(current["code"].upper())
                    replaced = False
                    for j in range(slot["current_idx"] + 1, len(slot["candidates"])):
                        candidate = slot["candidates"][j]
                        code_upper = candidate["code"].upper()
                        if code_upper in current_used or code_upper in taken_codes_all:
                            continue
                        old_cr = _credits_from_str(current.get("credits"))
                        new_cr = _credits_from_str(candidate.get("credits"))
                        current_total = sum(
                            _credits_from_str(slot_info["candidates"][slot_info["current_idx"]].get("credits"))
                            for slot_info in st.session_state.schedule_slots
                        )
                        if current_total - old_cr + new_cr <= st.session_state.schedule_target_credits:
                            _push_swap(idx, slot["current_idx"])
                            slot["current_idx"] = j
                            replaced = True
                            break
                    if not replaced:
                        st.info(f"No more eligible options for {current['code']} — {current['title']}.")
                    st.rerun()
            with cols[2]:
                if st.button("Lock" if not slot.get("locked") else "Unlock", key=f"lock_{slot['id']}"):
                    _toggle_lock(slot)
                    st.rerun()
            with cols[3]:
                pass
            new_total += _credits_from_str(current.get("credits"))
        st.session_state.schedule_planned_credits = new_total
        st.success(f"Planned credits: {new_total} / Target {st.session_state.schedule_target_credits}")

        csv_bytes = _export_schedule_csv(st.session_state.schedule_slots)
        st.download_button("Export schedule as CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        json_bytes = export_schedule_json(st.session_state.schedule_slots)
        st.download_button("Save schedule (JSON)", data=json_bytes, file_name="schedule.json", mime="application/json")
        upload = st.file_uploader("Load a saved schedule (JSON)", type=["json"], key="sched_loader")
        if upload is not None:
            try:
                st.session_state.schedule_slots = import_schedule_json(upload.read())
                st.session_state.schedule_planned_credits = sum(
                    _credits_from_str(slot["candidates"][slot["current_idx"]].get("credits"))
                    for slot in st.session_state.schedule_slots
                )
                st.rerun()
            except Exception as exc:
                st.warning(f"Could not load schedule: {exc}")


__all__ = [
    "LA_CATEGORY",
    "build_semester_schedule",
    "export_schedule_json",
    "import_schedule_json",
    "render_schedule_builder",
    "student_context_from_taken",
]
