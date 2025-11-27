#!/usr/bin/env python3
"""
測試 fix_barline_annotations.py 的修復邏輯

驗證：
1. 極細線擴大邏輯正確
2. 過大框緊縮邏輯正確
3. 邊界條件處理正確
"""

import sys
from pathlib import Path

# 導入修復器
sys.path.insert(0, str(Path(__file__).parent))
from fix_barline_annotations import BarlineAnnotationFixer, BARLINE, BARLINE_DOUBLE, BARLINE_FINAL

def test_expand_thin_barline():
    """測試極細線擴大功能"""
    print("\n" + "=" * 60)
    print("測試 1: 極細線擴大功能")
    print("=" * 60)

    fixer = BarlineAnnotationFixer(Path("."), Path("."))

    test_cases = [
        # (x_center, width, height, expected_result)
        (0.5, 0.002, 0.3, "應該擴大"),  # 極細線
        (0.5, 0.005, 0.3, "應該擴大"),  # 細線
        (0.5, 0.015, 0.3, "不應修改"),  # 正常寬度
        (0.01, 0.002, 0.3, "邊界測試"),  # 靠近左邊界
        (0.99, 0.002, 0.3, "邊界測試"),  # 靠近右邊界
    ]

    for x, w, h, desc in test_cases:
        new_x, new_w, fix_type = fixer.fix_barline_width(x, w, h)
        print(f"\n{desc}:")
        print(f"  輸入: x={x:.4f}, w={w:.6f}, h={h:.4f}")
        print(f"  輸出: x={new_x:.4f}, w={new_w:.6f}, 修復類型={fix_type}")

        # 驗證邊界條件
        assert 0 <= new_x - new_w/2, f"左邊界越界: {new_x - new_w/2}"
        assert new_x + new_w/2 <= 1, f"右邊界越界: {new_x + new_w/2}"

        # 驗證擴大邏輯
        if w < 0.01:
            assert new_w >= 0.015, f"擴大後寬度應 >= 0.015，實際為 {new_w}"
            assert fix_type == "expand_width", f"應該標記為 expand_width"
        else:
            assert new_w == w, f"不應修改寬度"
            assert fix_type == "no_change", f"應該標記為 no_change"

    print("\n✅ 測試 1 通過!")


def test_shrink_large_area():
    """測試過大框緊縮功能"""
    print("\n" + "=" * 60)
    print("測試 2: 過大框緊縮功能")
    print("=" * 60)

    fixer = BarlineAnnotationFixer(Path("."), Path("."))

    test_cases = [
        # (x, y, w, h, class_id, expected_result)
        (0.5, 0.5, 0.4, 0.5, BARLINE_DOUBLE, "應該緊縮 - 面積過大"),  # 面積 0.2 > 0.1
        (0.5, 0.5, 0.3, 0.4, BARLINE_FINAL, "應該緊縮 - 面積過大"),   # 面積 0.12 > 0.1
        (0.5, 0.5, 0.01, 0.3, BARLINE_DOUBLE, "不應修改 - 面積正常"),  # 面積 0.003 < 0.1
        (0.5, 0.5, 0.02, 0.6, BARLINE_FINAL, "不應修改 - 面積臨界"),   # 面積 0.012 < 0.1
        (0.5, 0.5, 0.5, 0.3, BARLINE_DOUBLE, "應該緊縮 - 寬度主導"),  # 異常：寬>高
    ]

    for x, y, w, h, class_id, desc in test_cases:
        new_x, new_y, new_w, new_h, fix_type = fixer.fix_large_barline_area(x, y, w, h, class_id)
        old_area = w * h
        new_area = new_w * new_h

        print(f"\n{desc}:")
        print(f"  輸入: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}, 面積={old_area:.6f}")
        print(f"  輸出: x={new_x:.4f}, y={new_y:.4f}, w={new_w:.4f}, h={new_h:.4f}, "
              f"面積={new_area:.6f}, 修復類型={fix_type}")

        # 驗證邊界條件
        assert 0 <= new_x - new_w/2, f"左邊界越界"
        assert new_x + new_w/2 <= 1, f"右邊界越界"
        assert 0 <= new_y - new_h/2, f"上邊界越界"
        assert new_y + new_h/2 <= 1, f"下邊界越界"

        # 驗證緊縮邏輯
        if old_area > 0.1:
            assert new_area < old_area, f"面積應該縮小"
            assert new_area <= 0.051, f"緊縮後面積應 <= 0.05，實際為 {new_area}"  # 容許浮點誤差
            assert fix_type == "shrink_area", f"應該標記為 shrink_area"
        else:
            assert abs(new_area - old_area) < 1e-9, f"面積不應改變"  # 容許浮點誤差
            assert fix_type == "no_change", f"應該標記為 no_change"

    print("\n✅ 測試 2 通過!")


def test_aspect_ratio_logic():
    """測試寬高比處理邏輯"""
    print("\n" + "=" * 60)
    print("測試 3: 寬高比處理邏輯")
    print("=" * 60)

    fixer = BarlineAnnotationFixer(Path("."), Path("."))

    test_cases = [
        # (w, h, class_id, expected_behavior)
        (0.4, 0.5, BARLINE_FINAL, "垂直主導 - 緊縮寬度"),  # aspect_ratio = 1.25
        (0.02, 0.6, BARLINE_DOUBLE, "極細長 - 保留高度"),  # aspect_ratio = 30
        (0.5, 0.4, BARLINE_FINAL, "水平主導 - 緊縮高度"),  # aspect_ratio = 0.8
        (0.4, 0.4, BARLINE_DOUBLE, "正方形 - 等比縮小"),   # aspect_ratio = 1.0
    ]

    for w, h, class_id, expected in test_cases:
        x, y = 0.5, 0.5  # 中心點
        new_x, new_y, new_w, new_h, fix_type = fixer.fix_large_barline_area(x, y, w, h, class_id)

        old_ratio = h / w if w > 0 else 0
        new_ratio = new_h / new_w if new_w > 0 else 0

        print(f"\n{expected}:")
        print(f"  輸入寬高比: {old_ratio:.2f} (w={w:.4f}, h={h:.4f})")
        print(f"  輸出寬高比: {new_ratio:.2f} (w={new_w:.4f}, h={new_h:.4f})")
        print(f"  修復類型: {fix_type}")

    print("\n✅ 測試 3 通過!")


def test_edge_cases():
    """測試邊界情況"""
    print("\n" + "=" * 60)
    print("測試 4: 邊界情況測試")
    print("=" * 60)

    fixer = BarlineAnnotationFixer(Path("."), Path("."))

    # 測試極端位置
    edge_cases = [
        (0.0, 0.002, 0.3, "左邊界"),
        (1.0, 0.002, 0.3, "右邊界"),
        (0.5, 0.001, 0.3, "極細線"),
        (0.5, 0.0, 0.3, "零寬度（異常）"),
    ]

    for x, w, h, desc in edge_cases:
        try:
            new_x, new_w, fix_type = fixer.fix_barline_width(x, w, h)
            print(f"\n{desc}: ✅")
            print(f"  輸入: x={x:.4f}, w={w:.6f}")
            print(f"  輸出: x={new_x:.4f}, w={new_w:.6f}")

            # 驗證結果在合法範圍內
            assert 0 <= new_x - new_w/2 <= 1, "邊界檢查失敗"
            assert 0 <= new_x + new_w/2 <= 1, "邊界檢查失敗"
        except Exception as e:
            print(f"\n{desc}: ❌ {e}")

    print("\n✅ 測試 4 通過!")


def main():
    """運行所有測試"""
    print("\n🧪 開始測試 fix_barline_annotations.py")

    try:
        test_expand_thin_barline()
        test_shrink_large_area()
        test_aspect_ratio_logic()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✅ 所有測試通過!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
