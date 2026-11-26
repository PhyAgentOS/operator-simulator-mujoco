"""Evaluate actual MuJoCo contacts without affecting physical state/control."""
import mujoco
import numpy as np
from forge_msgs import LocomotionCommand

from mujoco_sim.contact_evaluation import ContactEvaluation
from test_navigation_simulation import XML, make_base, step


def test_real_wall_contact_cumulative_evidence_survives_separation():
    model, data, base = make_base()
    evaluation = ContactEvaluation(model, data, base.body_id)
    for _ in range(3000):
        step(model, data, base, 1, LocomotionCommand(vx=.6, vy=0, wz=0))
        before = data.qpos.copy(), data.qvel.copy(), data.qfrc_applied.copy()
        evaluation.sample()
        for actual, expected in zip((data.qpos, data.qvel, data.qfrc_applied), before):
            np.testing.assert_array_equal(actual, expected)
    result = evaluation.snapshot()
    assert result['contact_physics_steps'] > 0
    assert result['current_contact_pair_count'] > 0
    assert result['pairs'][0]['obstacle_geom'] == 'wall'
    # Peak impact penetration is greater than settled contact penetration.
    assert 0 < result['max_penetration_m'] < .005
    assert 0 < result['first_contact_ns'] <= result['last_contact_ns'] <= 6_000_000_000
    for _ in range(500):
        step(model, data, base, 1, LocomotionCommand(vx=-.3, vy=0, wz=0))
        evaluation.sample()
    separated = evaluation.snapshot()
    assert separated['current_contact_pair_count'] == 0
    assert separated['contact_physics_steps'] >= result['contact_physics_steps']
    assert separated['first_contact_ns'] == result['first_contact_ns']
    evaluation.reset()
    assert evaluation.snapshot()['contact_physics_steps'] == 0
    assert evaluation.snapshot()['pairs'] == []


def test_hovering_stationary_base_has_no_false_contacts():
    model, data, base = make_base()
    evaluation = ContactEvaluation(model, data, base.body_id)
    for _ in range(100):
        step(model, data, base, 1)
        evaluation.sample()
    assert evaluation.snapshot()['contact_physics_steps'] == 0


def test_actual_floor_support_and_robot_internal_contacts_are_excluded():
    xml = '''<mujoco><worldbody>
    <geom name="floor" type="plane" size="3 3 .1"/>
    <body name="robot_root" pos="0 0 .19"><freejoint/>
      <geom name="base" type="sphere" size=".2"/>
      <body pos=".1 0 0"><joint name="arm" type="hinge"/><geom name="arm" type="sphere" size=".1"/></body>
    </body></worldbody><contact><pair geom1="base" geom2="arm"/></contact></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pairs = {(model.geom(c.geom1).name, model.geom(c.geom2).name) for c in data.contact[:data.ncon]}
    assert any('floor' in pair for pair in pairs)
    assert ('base', 'arm') in pairs or ('arm', 'base') in pairs
    evaluation = ContactEvaluation(model, data, model.body('robot_root').id)
    evaluation.sample()
    assert evaluation.snapshot()['contact_physics_steps'] == 0


def test_floor_side_is_not_blanket_excluded():
    xml = XML.replace('name="wall"', 'name="floor"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 1.61  # deliberate static overlap, evaluation only
    mujoco.mj_forward(model, data)
    evaluation = ContactEvaluation(model, data, model.body('robot_root').id)
    evaluation.sample()
    assert evaluation.snapshot()['contact_physics_steps'] == 1
    assert evaluation.snapshot()['pairs'][0]['obstacle_geom'] == 'floor'
