"""Archive regressions exercise content identity and persisted runtime composition."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from command_optimization.provenance import RELEASE_VERSION, source_identity


def _source_tree(root: Path) -> Path:
    (root / 'src' / 'example').mkdir(parents=True)
    (root / 'src' / 'example' / 'main.py').write_text('VALUE = 1\n')
    (root / 'pyproject.toml').write_text('[project]\nname = "example"\n')
    return root


def test_source_identity_is_git_free_relocatable_and_ignores_caches(tmp_path, monkeypatch):
    def forbid_process(*args, **kwargs):
        pytest.fail('source identity must not launch subprocesses')
    monkeypatch.setattr(subprocess, 'run', forbid_process)
    first = _source_tree(tmp_path / 'first')
    expected = source_identity(first)
    assert expected.release_version == RELEASE_VERSION
    assert len(expected.source_sha256) == 64
    (tmp_path / '.git').mkdir()
    second = tmp_path / 'elsewhere' / 'archive'
    shutil.copytree(first, second)
    for relative in ('src/example/__pycache__/main.pyc', 'src/example/.private/token',
                     'outputs/report.json', '.git/HEAD', '.env', 'src/example/main.py~'):
        path = second / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('ignored')
    assert source_identity(second) == expected
    (second / 'src/example/main.py').write_text('VALUE = 2\n')
    assert source_identity(second) != expected


def test_source_identity_hashes_relative_names_and_rejects_missing_source(tmp_path):
    root = _source_tree(tmp_path / 'source')
    before = source_identity(root)
    (root / 'src/example/main.py').rename(root / 'src/example/other.py')
    assert source_identity(root) != before
    with pytest.raises(FileNotFoundError, match='source'):
        source_identity(tmp_path / 'missing')


@pytest.fixture
def archive_runtime(tmp_path, monkeypatch):
    """Use real config, artifact verification, records and metrics; fake GPU work."""
    import json
    from types import SimpleNamespace
    import numpy as np
    import yaml
    import command_optimization.run as run
    import command_optimization.batch as batch
    from bfm_adapters import MotionCommand, instance_config_from_mapping
    from command_optimization.artifacts import _tree_stats, sha256_file
    from command_optimization.records import ExecutionTrace, PlanningDiagnostics
    from command_optimization.simulation.measurements import ControlStepMeasurements, MeasurementPrimitive

    repo = Path(__file__).resolve().parents[2]
    root = _source_tree(tmp_path / 'archive')
    shutil.copytree(repo / 'configs', root / 'configs')
    resolved, _ = run.load_experiment(repo / 'tests/fixtures/experiments/heft_smoke.yaml')
    resolved['reference'] = {'motion_id': 'demo', 'dataset_root': str(root / 'data'), 'clip_duration_seconds': 0.08}
    resolved['output'] = {'formats': [], 'trajectories': []}
    resolved['robot']['artifact'] = 'robot'
    resolved['robot']['entrypoint'] = 'g1.xml'
    resolved['bfm']['artifacts']['policy'] = 'policy'
    experiment = root / 'experiment.yaml'
    experiment.write_text(yaml.safe_dump(resolved))
    (root / 'data/motions').mkdir(parents=True)
    (root / 'data/manifest.json').write_text('{}')
    (root / 'data/motions/demo.npz').write_bytes(b'reference-input')
    (root / 'robot').mkdir()
    (root / 'robot/g1.xml').write_text('<mujoco/>')
    policy = root / 'policy.onnx'
    policy.write_bytes(b'policy-input')
    count, size, digest = _tree_stats(root / 'robot')
    (root / 'artifacts').mkdir()
    (root / 'artifacts/lock.v1.json').write_text(json.dumps({'schema_version': 1, 'artifacts': {
        'policy': {'kind': 'file', 'path': 'policy.onnx', 'sha256': sha256_file(policy), 'bytes': policy.stat().st_size},
        'robot': {'kind': 'tar.gz', 'sha256': '0' * 64, 'bytes': 0, 'tree_root': 'robot',
                  'tree_files': count, 'tree_bytes': size, 'tree_sha256': digest},
    }}))
    monkeypatch.setattr(run, 'default_repo_root', lambda: root)
    monkeypatch.setattr(batch, 'default_repo_root', lambda: root)
    monkeypatch.setattr(run, '_hardware_identity', lambda _: {'available': True, 'gpu_uuid': 'GPU-test'})
    monkeypatch.setattr(run, 'version', lambda _: 'test-version')
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: pytest.fail('archive runtime must not invoke Git or other external processes'))
    bfm = instance_config_from_mapping(resolved['bfm'])
    bodies = tuple(resolved['motion_tracking']['body_names'])
    joints = bfm.command_joint_names
    orientation = np.zeros((1, 5, len(bodies), 4))
    orientation[..., 0] = 1
    vectors = np.zeros((1, 5, len(bodies), 3))
    reference = MotionCommand(
        query_offsets=tuple(range(5)), joint_names=joints, body_names=bodies,
        joint_position=np.zeros((1, 5, len(joints))), joint_velocity=np.zeros((1, 5, len(joints))),
        body_position_world=vectors.copy(), body_orientation_world=orientation,
        body_linear_velocity_world=vectors.copy(), body_angular_velocity_world=vectors.copy(),
    )
    monkeypatch.setattr(run, 'load_reference_motion', lambda *args, **kwargs: reference)

    def physical(command, period):
        qpos = np.zeros((1, len(command.query_offsets), 7 + len(joints)))
        qpos[..., 3] = 1
        return qpos, np.zeros((1, len(command.query_offsets), 6 + len(joints)))

    def execute(state, command, **kwargs):
        qpos, qvel = physical(command, bfm.control_period)
        measured = ControlStepMeasurements(joints, bodies, {
            primitive: getattr(command, primitive.value).copy() for primitive in MeasurementPrimitive
        })
        return SimpleNamespace(qpos=qpos, qvel=qvel, measurements=measured)

    class Runtime:
        implementation = 'device'
        def __init__(self, **kwargs):
            pass
        def world(self, _):
            return SimpleNamespace(body_names=bodies, physical_trajectory=physical)
        def executor(self, _):
            return SimpleNamespace(execute=execute)
        def close(self):
            pass

    def optimize(**kwargs):
        command = kwargs['reference_future']
        result = execute(kwargs['initial_execution_state'], command)
        trace = ExecutionTrace(
            timeline=bfm.control_period * np.arange(1, 5), reference_motion=command,
            applied_command=command, executed_trajectory=result.measurements,
            qpos=result.qpos[0], qvel=result.qvel[0], valid=np.ones(4, dtype=bool), root_body_name='pelvis',
        )
        diagnostics = PlanningDiagnostics(
            update_step=0, selected_source='best_observed', selected_candidate_id='best', selected_cost=0.0,
            best_observed_candidate_id='best', best_observed_cost=0.0, proposal_candidate_id='best', proposal_cost=0.0,
            cost_quantiles=np.zeros((1, 5)), candidate_deformation={'joint_rmse_rad': 0.0},
            candidate_evaluations=5, scenario_count=1, scenario_digest=None, logical_rollouts=5, physical_rollouts=5,
            penalized_candidate_evaluations=0, invalid_scenario_rollouts=0, update_wall_seconds=0.01,
        )
        return SimpleNamespace(trace=trace, planning=(diagnostics,))

    monkeypatch.setattr(run, '_RolloutRuntimePool', Runtime)
    monkeypatch.setattr(run, 'run_command_optimization', optimize)
    return SimpleNamespace(root=root, experiment=experiment, policy=policy, run=run, batch=batch)


def test_archive_run_records_and_renders_without_git(archive_runtime, monkeypatch):
    import json
    import numpy as np
    import command_optimization.visualization.render as rendering
    runtime = archive_runtime
    result = runtime.run.run_experiment(runtime.experiment, output_dir=runtime.root / 'outputs/single')
    provenance = json.loads((result.output_dir / 'provenance.json').read_text())
    assert provenance['release_version'] == RELEASE_VERSION
    assert provenance['source_sha256'] == source_identity(runtime.root).source_sha256
    assert not {'source_commit', 'source_dirty', 'mjcf_path'} & provenance.keys()
    report = json.loads((result.output_dir / 'report.json').read_text())
    assert report['baseline']['success'] and report['optimized']['success']
    assert report['optimized']['total_steps'] == 4
    assert len(report['identities']['checkpoint_digest']) == 64
    monkeypatch.setattr(rendering, '_render_frames', lambda *args: iter([np.zeros((4, 8, 3), dtype=np.uint8)]))
    monkeypatch.setattr(rendering, '_encode_mp4', lambda frames, path, **kwargs: path.write_bytes(b'encoded-frames'))
    runtime.run._render_experiment(result.output_dir)
    rendered = json.loads((result.output_dir / 'render_provenance.json').read_text())
    assert rendered['source_sha256'] == provenance['source_sha256']
    assert rendered['release_version'] == provenance['release_version']


def test_archive_batch_executes_actual_run_and_aggregates_files(archive_runtime):
    import json
    import yaml
    runtime = archive_runtime
    path = runtime.root / 'batch.yaml'
    path.write_text(yaml.safe_dump({
        'schema_version': 3, 'name': 'archive', 'experiment_refs': ['experiment.yaml'],
        'budgets': [{'id': 'small', 'num_candidates': 5, 'num_iterations': 1}],
        'optimizer_variants': [{'id': 'weighted_elites', 'ref': 'optimizers/weighted_elites.yaml', 'overrides': {}}],
        'seeds': [0, 1], 'rollout_batch_ceiling': 5,
    }))
    result = runtime.batch.run_experiment_batch(path, execute_experiment=runtime.run.run_experiment,
                                              output_dir=runtime.root / 'outputs/batch')
    summary = json.loads((result.output_dir / 'experiment_batch_summary.json').read_text())
    assert result.success, summary
    assert result.cells == 2
    assert summary['groups'][0]['n'] == summary['groups'][0]['successes'] == 2
    assert summary['experiment_batch']['source_sha256'] == source_identity(runtime.root).source_sha256
    for cell in summary['cells']:
        assert (result.output_dir / cell['output'] / 'paired_traces.npz').is_file()
        assert cell['baseline']['total_steps'] == cell['optimized']['total_steps'] == 4


@pytest.mark.parametrize('broken', ['missing_policy', 'corrupt_policy', 'missing_motion'])
def test_archive_asset_failures_remain_errors(archive_runtime, broken):
    runtime = archive_runtime
    if broken == 'missing_policy':
        runtime.policy.unlink()
    elif broken == 'corrupt_policy':
        runtime.policy.write_bytes(b'corrupt')
    else:
        (runtime.root / 'data/motions/demo.npz').unlink()
    with pytest.raises((FileNotFoundError, ValueError), match='artifact|motion'):
        runtime.run.run_experiment(runtime.experiment, output_dir=runtime.root / 'outputs/failed')
    assert not (runtime.root / 'outputs/failed/report.json').exists()
