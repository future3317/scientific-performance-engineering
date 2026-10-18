from experiments.external_real.skillsbench_transfer import (
    enumerate_skills,
    materialize_no_skill_task,
    materialize_native_skill_task,
    materialize_transferred_skill_task,
    materialize_sb1_smoke,
)


def test_no_skill_materialization_removes_skill_tree(tmp_path):
    source = tmp_path / "source"
    (source / "environment" / "skills" / "demo").mkdir(parents=True)
    (source / "environment" / "skills" / "demo" / "SKILL.md").write_text("secret", encoding="utf-8")
    (source / "task.md").write_text("task", encoding="utf-8")

    destination = tmp_path / "sanitized"
    materialize_no_skill_task(source, destination)

    assert not (destination / "environment" / "skills").exists()
    assert not list(destination.rglob("SKILL.md"))


def test_skill_source_task_id_is_read_from_task_path(tmp_path):
    benchmark = tmp_path / "benchmark"
    task = benchmark / "tasks" / "task-001"
    skill = task / "environment" / "skills" / "transferable"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: transferable\ndescription: reusable workflow\n---\nUse the workflow.\n",
        encoding="utf-8",
    )

    rows = enumerate_skills(
        benchmark,
        [{
            "task_id": "task-001",
            "category": "science",
            "difficulty": "easy",
            "task_type": [],
            "modality": [],
            "interface": [],
            "skill_type": [],
            "tags": [],
        }],
    )

    assert rows[0]["source_task_id"] == "task-001"


def test_native_materialization_keeps_only_task_owned_skills(tmp_path):
    source = tmp_path / "source"
    skill = source / "environment" / "skills" / "native"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("native", encoding="utf-8")
    (source / "task.md").write_text("task", encoding="utf-8")

    destination = tmp_path / "native"
    materialize_native_skill_task(source, destination)

    assert (destination / "environment" / "skills" / "native" / "SKILL.md").read_text() == "native"
    assert not list(destination.rglob("transferred-SKILL.md"))


def test_transferred_materialization_injects_one_frozen_skill(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    (source / "task.md").write_text("task", encoding="utf-8")
    source_skill = tmp_path / "source-skill"
    source_skill.mkdir()
    (source_skill / "SKILL.md").write_text("frozen skill", encoding="utf-8")

    destination = tmp_path / "transferred"
    materialize_transferred_skill_task(source, source_skill, "frozen", destination)

    skill_files = list((destination / "environment" / "skills").rglob("SKILL.md"))
    assert [path.relative_to(destination).as_posix() for path in skill_files] == [
        "environment/skills/frozen/SKILL.md"
    ]
    assert skill_files[0].read_text() == "frozen skill"


def test_sb1_smoke_materialization_records_sanitized_conditions(tmp_path):
    benchmark = tmp_path / "benchmark"
    task = benchmark / "tasks" / "task-001"
    skill = task / "environment" / "skills" / "native"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("native", encoding="utf-8")
    (task / "task.md").write_text("task", encoding="utf-8")
    manifest = tmp_path / "smoke.json"
    manifest.write_text(
        '{"benchmark_release":"v1.1","conditions":["oracle","no_skill","native_skill"],'
        '"tasks":[{"task_id":"task-001"}]}',
        encoding="utf-8",
    )

    receipt = materialize_sb1_smoke(benchmark, manifest, tmp_path / "out")

    assert receipt["provider_calls"] == 0
    assert receipt["tasks"] == ["task-001"]
    assert not list((tmp_path / "out" / "no_skill" / "task-001").rglob("SKILL.md"))
    assert list((tmp_path / "out" / "native_skill" / "task-001").rglob("SKILL.md"))
