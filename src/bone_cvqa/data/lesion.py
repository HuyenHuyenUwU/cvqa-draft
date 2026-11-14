def missing_lesion():
    """Nhãn tổn thương vắng mặt được đánh dấu tường minh (không điền 0 làm target)."""
    return {"status": "missing", "box_xyxy_norm": None, "format": "xyxy_normalized", "source": None}
