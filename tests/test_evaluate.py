"""
Tests for the evaluation module — IoU math and the precision/recall/F1
bookkeeping. Doesn't require an actual labeled test set or a loaded model.
Run with: pytest tests/
"""

from src.evaluate import EvalStats, _iou, _load_ground_truth


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert _iou(box, box) == 1.0


def test_iou_no_overlap_is_zero():
    assert _iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0


def test_iou_partial_overlap_between_zero_and_one():
    iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < iou < 1.0


def test_load_ground_truth_missing_file_returns_empty():
    from pathlib import Path

    result = _load_ground_truth(Path("/tmp/does_not_exist_12345.txt"), 640, 480, {0: "person"})
    assert result == []


def test_load_ground_truth_parses_yolo_format(tmp_path):
    label_file = tmp_path / "img001.txt"
    # class 0, centered box, half the image width/height
    label_file.write_text("0 0.5 0.5 0.5 0.5\n")

    ground_truths = _load_ground_truth(label_file, img_w=640, img_h=480, class_names={0: "person"})

    assert len(ground_truths) == 1
    gt = ground_truths[0]
    assert gt["class_name"] == "person"
    assert gt["matched"] is False
    x1, y1, x2, y2 = gt["box"]
    assert round(x2 - x1) == 320  # half of 640
    assert round(y2 - y1) == 240  # half of 480


def test_eval_stats_precision_recall_f1():
    stats = EvalStats()
    stats.add_tp("person")
    stats.add_fp("person")
    stats.add_fn("person")

    rows = stats.report_rows()
    person_row = next(r for r in rows if r[0] == "person")

    # TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, f1=0.5
    assert person_row[1:4] == [1, 1, 1]
    assert person_row[4] == 0.5
    assert person_row[5] == 0.5
    assert person_row[6] == 0.5


def test_eval_stats_overall_row_aggregates_all_classes():
    stats = EvalStats()
    stats.add_tp("person")
    stats.add_tp("chair")
    stats.add_fp("chair")

    rows = stats.report_rows()
    overall_row = next(r for r in rows if r[0] == "OVERALL")

    assert overall_row[1] == 2  # total TP
    assert overall_row[2] == 1  # total FP
    assert overall_row[3] == 0  # total FN


def test_eval_stats_perfect_precision_when_no_false_positives():
    stats = EvalStats()
    stats.add_tp("person")
    rows = stats.report_rows()
    person_row = next(r for r in rows if r[0] == "person")
    assert person_row[4] == 1.0  # precision
