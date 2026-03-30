import json

from evaluation import task_planner


SAMPLE_BDDL = """(define (problem LIBERO_Coffee_Table_Manipulation)
  (:domain robosuite)
  (:language Organize selected food items into the white_storage_box)
  (:fixtures
    coffee_table - coffee_table
    white_storage_box_1 - white_storage_box
  )
  (:objects
    black_book_1 - black_book
    butter_1 - butter
    cream_cheese_1 - cream_cheese
    popcorn_1 - popcorn
  )
  (:obj_of_interest
    white_storage_box_1
    cream_cheese_1
    popcorn_1
    butter_1
  )
  (:init
    (On white_storage_box_1 coffee_table_white_storage_box_init_region)
    (On cream_cheese_1 coffee_table_cream_cheese_init_region)
    (On popcorn_1 coffee_table_popcorn_init_region)
    (On butter_1 coffee_table_butter_init_region)
    (On black_book_1 coffee_table_black_book_init_region)
  )
  (:goal
    (And
      (On white_storage_box_1 coffee_table_white_storage_box_init_region)
      (In cream_cheese_1 white_storage_box_1_bottom_side)
      (In popcorn_1 white_storage_box_1_right_side)
      (In butter_1 white_storage_box_1_left_side)
    )
  )
)
"""


def test_task_plan_bootstrap_writes_expected_json(tmp_path, monkeypatch):
    bddl_path = tmp_path / "sample_task.bddl"
    bddl_path.write_text(SAMPLE_BDDL, encoding="utf-8")

    def fake_get_robosuite_parse_problem():
        def _parse_problem(_):
            return {
                "fixtures": {
                    "coffee_table": ["coffee_table"],
                    "white_storage_box": ["white_storage_box_1"],
                },
                "objects": {
                    "black_book": ["black_book_1"],
                    "butter": ["butter_1"],
                    "cream_cheese": ["cream_cheese_1"],
                    "popcorn": ["popcorn_1"],
                },
                "obj_of_interest": [
                    "white_storage_box_1",
                    "cream_cheese_1",
                    "popcorn_1",
                    "butter_1",
                ],
                "initial_state": [
                    ["On", "white_storage_box_1", "coffee_table_white_storage_box_init_region"],
                    ["On", "cream_cheese_1", "coffee_table_cream_cheese_init_region"],
                    ["On", "popcorn_1", "coffee_table_popcorn_init_region"],
                    ["On", "butter_1", "coffee_table_butter_init_region"],
                ],
                "goal_state": [
                    ["On", "white_storage_box_1", "coffee_table_white_storage_box_init_region"],
                    ["In", "cream_cheese_1", "white_storage_box_1_bottom_side"],
                    ["In", "popcorn_1", "white_storage_box_1_right_side"],
                    ["In", "butter_1", "white_storage_box_1_left_side"],
                ],
                "language_instruction": "Organize selected food items into the white_storage_box",
            }

        return _parse_problem

    monkeypatch.setattr(task_planner, "_get_robosuite_parse_problem", fake_get_robosuite_parse_problem)

    text_info = task_planner.parse_bddl_text(bddl_path)
    metadata = task_planner.parse_bddl_metadata(bddl_path)
    plan = task_planner.build_task_plan_from_bddl(text_info, metadata)
    output_path = task_planner.write_task_plan_json(plan, tmp_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "vlm_planning_tree.json"
    assert written.get("task", written.get("task_id")) == "LIBERO_Coffee_Table_Manipulation"
    assert written["language_instruction"] == "Organize selected food items into the white_storage_box"
    assert written["formal_goal"] == (
        "(And (On white_storage_box_1 coffee_table_white_storage_box_init_region) "
        "(In cream_cheese_1 white_storage_box_1_bottom_side) "
        "(In popcorn_1 white_storage_box_1_right_side) "
        "(In butter_1 white_storage_box_1_left_side))"
    )
    assert written["root"]["goal_summary"] == [
        "white_storage_box_1 on coffee_table_white_storage_box_init_region",
        "cream_cheese_1 in white_storage_box_1_bottom_side",
        "popcorn_1 in white_storage_box_1_right_side",
        "butter_1 in white_storage_box_1_left_side",
    ]
