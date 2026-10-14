from scripts.reproduce_final_verification_20261014 import steps


def test_replay_select_then_test_and_isolates_published_results(tmp_path):
    plan = steps(tmp_path, tmp_path / "isolated", {"2", "5", "6"})
    names = [s["name"] for s in plan]
    assert names.index("task5-neural-select") < names.index("task5-neural-test")
    for step in plan:
        assert "--publish" not in step["argv"]
        for option in ("--out", "--results", "--results-dir"):
            if option in step["argv"]:
                assert "isolated" in step["argv"][step["argv"].index(option) + 1]
    assert len(plan) == 10


def test_task_filter_keeps_data_cohort_and_pairing(tmp_path):
    plan = steps(tmp_path, tmp_path / "fresh", {"2"})
    assert len(plan) == 3
    assert "data" in plan[0]["argv"][plan[0]["argv"].index("--annotations") + 1]
    assert plan[1]["argv"][plan[1]["argv"].index("--base") + 1] == str(tmp_path / "fresh/task2")
