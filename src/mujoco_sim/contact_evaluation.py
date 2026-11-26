"""Passive robot/world contact accounting; never changes state or control."""
from __future__ import annotations

import mujoco


class ContactEvaluation:
    def __init__(self, model, data, robot_body_id):
        self.model, self.data = model, data
        robot_bodies = {robot_body_id}
        for body in range(robot_body_id + 1, model.nbody):
            if int(model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        self.robot_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies}
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor >= 0 and model.geom_bodyid[self.floor] != 0:
            self.floor = -1
        self.reset()

    def reset(self):
        self.sampled_steps = 0
        self.contact_steps = 0
        self.pair_events = 0
        self.first_contact_ns = None
        self.last_contact_ns = None
        self.capture_timestamp_ns = 0
        self.active_pairs = set()
        self.pairs = {}
        self.max_penetration = 0.0

    def sample(self):
        self.sampled_steps += 1
        stamp = round(float(self.data.time) * 1e9)
        self.capture_timestamp_ns = stamp
        active = {}
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            # Positive-distance contact candidates are not physical overlap.
            if contact.dist > 0 or (a in self.robot_geoms) == (b in self.robot_geoms):
                continue
            robot, obstacle = (a, b) if a in self.robot_geoms else (b, a)
            # Ignore only vertical support on the explicitly named world floor;
            # its side faces remain obstacles. No blanket plane/box exclusion.
            if obstacle == self.floor and abs(float(contact.frame[2])) >= 0.95:
                continue
            pair = (robot, obstacle)
            active[pair] = max(active.get(pair, 0.0), max(0.0, -float(contact.dist)))
        if active:
            self.contact_steps += 1
            if self.first_contact_ns is None:
                self.first_contact_ns = stamp
            self.last_contact_ns = stamp
        for pair, depth in active.items():
            if pair not in self.active_pairs:
                self.pair_events += 1
            if pair not in self.pairs:
                robot, obstacle = pair
                self.pairs[pair] = {
                    "robot_geom": self.model.geom(robot).name or f"geom:{robot}",
                    "obstacle_geom": self.model.geom(obstacle).name or f"geom:{obstacle}",
                    "first_contact_ns": stamp, "contact_steps": 0, "max_penetration_m": 0.0,
                }
            entry = self.pairs[pair]
            entry["last_contact_ns"] = stamp
            entry["contact_steps"] += 1
            entry["max_penetration_m"] = max(entry["max_penetration_m"], depth)
            self.max_penetration = max(self.max_penetration, depth)
        self.active_pairs = set(active)

    def snapshot(self):
        return {
            "schema_version": 1, "evaluation_only": True,
            "source": "mujoco_post_step_contacts", "clock_domain": "simulation",
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "sampled_physics_steps": self.sampled_steps,
            "contact_physics_steps": self.contact_steps,
            "pair_contact_events": self.pair_events,
            "current_contact_pair_count": len(self.active_pairs),
            "first_contact_ns": self.first_contact_ns, "last_contact_ns": self.last_contact_ns,
            "max_penetration_m": self.max_penetration,
            "pairs": [dict(self.pairs[pair]) for pair in sorted(self.pairs)],
            "exclusions": "same-robot contacts and vertical support contacts on world geom named floor",
        }
