# Schedule builder UI for AUIBPT

import streamlit as st
from typing import List, Dict, Set

from ..core.schedule import (
    build_semester_schedule, _rebuild_pools, _export_schedule_csv, 
    export_schedule_json, import_schedule_json, _auto_top_up,
    _toggle_lock, _push_swap, _undo_swap
)
from ..utils.constants import MAJOR_MAP, DEGREE_TOTAL, LA_CATEGORY
from ..utils.helpers import _credits_from_str, _credits_completed

def render_schedule_builder(rows_all, vs, bm25):
    """Render the schedule builder interface."""
    st.markdown("### Schedule Builder")
    
    if "schedule_difficulty" not in st.session_state:
        st.session_state.schedule_difficulty = "Medium"
    
    difficulty_choice = st.radio(
        "Semester difficulty",
        options=["Easy", "Medium", "Hard"],
        index=["Easy","Medium","Hard"].index(st.session_state.schedule_difficulty),
        horizontal=True,
        help=("Easy increases Liberal Arts weight. "
              "Medium uses 3:1 (Major:Liberal). "
              "Hard increases Major weight."),
    )
    st.session_state.schedule_difficulty = difficulty_choice

    try:
        options = list(MAJOR_MAP.keys())
    except Exception:
        options = ["CS"]
    if not options:
        options = ["CS"]
    
    try:
        default_idx = options.index(st.session_state.schedule_major_key) if "schedule_major_key" in st.session_state else 0
    except ValueError:
        default_idx = 0
    
    major_key = st.selectbox("Major / program", options, index=default_idx)
    st.session_state.schedule_major_key = major_key
    
    target_credits = st.slider("Target credits", 9, 21, st.session_state.schedule_target_credits, 1)
    st.session_state.schedule_target_credits = target_credits

    picker_scope = st.radio("Completed-course picker scope:", ["Major only", "Liberal Arts only", "Both"], horizontal=True)

    major_prefixes = MAJOR_MAP[major_key]["prefixes"]
    major_only_rows = [r for r in rows_all if r["code"].upper().startswith(major_prefixes)]
    la_only_rows    = [r for r in rows_all if r["code"].upper() in LA_CATEGORY]

    if picker_scope == "Major only":
        picker_rows = major_only_rows
    elif picker_scope == "Liberal Arts only":
        picker_rows = la_only_rows
    else:
        seen = set()
        picker_rows = []
        for r in major_only_rows + la_only_rows:
            cu = r["code"].upper()
            if cu not in seen:
                picker_rows.append(r); seen.add(cu)

    labels = [f"{r['code']} — {r['title']}" for r in picker_rows]
    label_to_code = {f"{r['code']} — {r['title']}": r["code"].upper() for r in picker_rows}
    visible_codes = set(label_to_code.values())

    preselected_labels = [lbl for lbl, code in label_to_code.items() if code in st.session_state.completed_codes_all]
    picked_labels = st.multiselect("I have completed:", labels, default=preselected_labels, key="completed_picker")
    picked_visible_codes = {label_to_code[lbl] for lbl in picked_labels}

    hidden_kept = st.session_state.completed_codes_all - visible_codes
    st.session_state.completed_codes_all = hidden_kept | picked_visible_codes
    taken_codes_all = set(st.session_state.completed_codes_all)

    completed_credits = _credits_completed(taken_codes_all, rows_all)
    degree_total = DEGREE_TOTAL[major_key]
    st.caption(f"Progress: {completed_credits} / {degree_total} credits • Target this term: {target_credits}")

    col_build_a, col_build_b, col_build_c, col_build_d, col_build_e = st.columns([0.35,0.2,0.2,0.15,0.1])
    with col_build_a:
        build_btn = st.button("Build schedule", use_container_width=True)
    with col_build_b:
        reset_btn = st.button("Reset", use_container_width=True)
    with col_build_c:
        topup_btn = st.button("Auto top-up", use_container_width=True, disabled=not st.session_state.schedule_slots)
    with col_build_d:
        undo_btn = st.button("Undo swap", use_container_width=True, disabled=not st.session_state.swap_history)
    with col_build_e:
        if st.button("Close", use_container_width=True):
            st.session_state.show_schedule = False
            st.rerun()

    if reset_btn:
        st.session_state.schedule_slots = []
        st.session_state.schedule_planned_credits = 0
        st.rerun()

    if build_btn:
        schedule, la_counts, la_remain, planned_credits = build_semester_schedule(
            major_key=major_key,
            target_credits=target_credits,
            taken_codes=taken_codes_all,
            rows_all=rows_all,
            difficulty=st.session_state.get("schedule_difficulty", "Medium")
        )
        la_pool, major_pool = _rebuild_pools(major_key, taken_codes_all, rows_all)

        slots = []
        used = set(taken_codes_all)
        for idx, c in enumerate(schedule):
            if c["code"].upper() in LA_CATEGORY:
                origin = f"LA:{LA_CATEGORY[c['code'].upper()]}"
                pool = la_pool.get(LA_CATEGORY[c["code"].upper()], [])
            else:
                origin = "Major"
                pool = major_pool

            candidates = []
            seen_codes = set()
            candidates.append(c); seen_codes.add(c["code"].upper())
            for r in pool:
                cu = r["code"].upper()
                if cu not in seen_codes and cu not in used:
                    candidates.append(r); seen_codes.add(cu)

            slots.append({
                "id": f"{origin}-{idx}",
                "origin": origin,
                "candidates": candidates,
                "current_idx": 0,
                "locked": False,
            })
            used.add(c["code"].upper())

        st.session_state.schedule_slots = slots
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][0].get("credits")) for s in slots)

    if topup_btn and st.session_state.schedule_slots:
        _auto_top_up(major_key, target_credits, taken_codes_all, st.session_state.schedule_slots, rows_all)
        st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
        st.rerun()

    if undo_btn:
        if _undo_swap(st.session_state.swap_history, st.session_state.schedule_slots):
            st.rerun()
        else:
            st.info("Nothing to undo.")

    if st.session_state.schedule_slots:
        st.markdown("### Suggested schedule")
        new_total = 0

        for i, slot in enumerate(st.session_state.schedule_slots):
            cur = slot["candidates"][slot["current_idx"]]
            pr = cur.get("prereqs") or "None/Unknown"
            cr = cur.get("credits") or "Unknown"

            cols = st.columns([0.64, 0.12, 0.12, 0.12])
            with cols[0]:
                st.markdown(f"**{cur['code']} — {cur['title']}**  \nCategory: {slot['origin']} • Credits: {cr} • Prereqs: {pr}  \nStatus: {'Locked' if slot.get('locked') else 'Unlocked'}")
                why_bits = []
                if cur["code"].upper() in LA_CATEGORY:
                    why_bits.append(f"meets {LA_CATEGORY[cur['code'].upper()]}")
                else:
                    why_bits.append("major requirement/elective")
                
                from ..utils.helpers import _parse_prereq_codes
                reqs = _parse_prereq_codes(cur.get("prereqs",""))
                if not reqs:
                    why_bits.append("no explicit prerequisites")
                else:
                    if all(rc in taken_codes_all for rc in reqs):
                        why_bits.append("prerequisites satisfied")
                    else:
                        why_bits.append("prerequisites satisfied during planning")
                st.caption("Why this? " + " • ".join(why_bits))

            with cols[1]:
                if st.button(
                    "Swap",
                    help="Replace with the next eligible option",
                    key=f"swap_{slot['id']}",
                    disabled=bool(slot.get("locked", False)),
                ):
                    current_used = {c["candidates"][c["current_idx"]]["code"].upper() for c in st.session_state.schedule_slots}
                    current_used.discard(cur["code"].upper())

                    replaced = False
                    for j in range(slot["current_idx"] + 1, len(slot["candidates"])):
                        cand = slot["candidates"][j]
                        code_u = cand["code"].upper()
                        if code_u in current_used or code_u in taken_codes_all:
                            continue
                        old_cr = _credits_from_str(cur.get("credits"))
                        new_cr = _credits_from_str(cand.get("credits"))
                        current_total = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                        if current_total - old_cr + new_cr <= st.session_state.schedule_target_credits:
                            _push_swap(i, slot["current_idx"], st.session_state.swap_history)
                            slot["current_idx"] = j
                            replaced = True
                            break
                    if not replaced:
                        st.info(f"No more eligible options left for {cur['code']} — {cur['title']} in {slot['origin']}.")
                    st.rerun()

            with cols[2]:
                if st.button("Lock" if not slot.get("locked") else "Unlock", key=f"lock_{slot['id']}"):
                    _toggle_lock(slot)
                    st.rerun()
            with cols[3]:
                pass

            new_total += _credits_from_str(cur.get("credits"))

        st.session_state.schedule_planned_credits = new_total
        st.success(f"Planned credits: {new_total} / Target {st.session_state.schedule_target_credits}")

        csv_bytes = _export_schedule_csv(st.session_state.schedule_slots)
        st.download_button("Export schedule as CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        json_bytes = export_schedule_json(st.session_state.schedule_slots)
        st.download_button("Save schedule (JSON)", data=json_bytes, file_name="schedule.json", mime="application/json")
        
        up = st.file_uploader("Load a saved schedule (JSON)", type=["json"], key="sched_loader")
        if up is not None:
            try:
                st.session_state.schedule_slots = import_schedule_json(up.read())
                st.session_state.schedule_planned_credits = sum(_credits_from_str(s["candidates"][s["current_idx"]].get("credits")) for s in st.session_state.schedule_slots)
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load schedule: {e}")
