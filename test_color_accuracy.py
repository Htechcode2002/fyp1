import cv2
import numpy as np
import sys
sys.path.append('src')
from core.detection import VideoDetector

# Create detector instance
detector = VideoDetector()

# Test cases: (BGR color, expected color name)
test_cases = [
    # Basic colors
    ([0, 0, 255], "Red", "Pure Red"),
    ([0, 0, 150], "Dark Red", "Dark Red"),
    ([0, 255, 0], "Green", "Pure Green"),
    ([0, 150, 0], "Dark Green", "Dark Green"),
    ([255, 0, 0], "Blue", "Pure Blue"),
    ([150, 0, 0], "Navy Blue", "Navy Blue"),
    ([0, 255, 255], "Yellow", "Pure Yellow"),
    ([255, 255, 255], "White", "Pure White"),
    ([0, 0, 0], "Black", "Pure Black"),
    ([128, 128, 128], "Gray", "Medium Gray"),

    # Complex colors
    ([20, 100, 255], "Orange", "Orange"),
    ([50, 50, 150], "Brown", "Brown"),
    ([180, 105, 255], "Pink", "Pink"),
    ([220, 0, 220], "Purple", "Purple"),
    ([255, 255, 0], "Cyan", "Cyan"),
    ([50, 200, 200], "Yellow Green", "Yellow-Green"),
    ([200, 150, 100], "Light Blue", "Light Blue"),

    # Realistic clothing colors
    ([40, 40, 180], "Red", "Red Shirt"),
    ([60, 80, 60], "Dark Green", "Dark Green Jacket"),
    ([140, 100, 40], "Navy Blue", "Navy Blue Jeans"),
    ([220, 220, 220], "White", "White T-Shirt"),
    ([30, 30, 30], "Black", "Black Hoodie"),
    ([100, 100, 200], "Orange", "Orange Vest"),
]

print("=" * 80)
print("COLOR DETECTION ACCURACY TEST")
print("=" * 80)
print()

correct = 0
total = len(test_cases)

results = []

for bgr, expected, description in test_cases:
    # Create a test ROI (100x100 pixels of solid color)
    roi = np.zeros((100, 100, 3), dtype=np.uint8)
    roi[:, :] = bgr

    # Add realistic noise (lighting variations, shadows)
    noise = np.random.randint(-15, 15, roi.shape, dtype=np.int16)
    roi = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Detect color
    detected = detector.get_dominant_color(roi)

    # Check if correct
    is_correct = (detected == expected)
    if is_correct:
        correct += 1
        status = "✓"
    else:
        status = "✗"

    results.append((description, bgr, expected, detected, is_correct))
    print(f"{status} {description:20} | Expected: {expected:15} | Got: {detected:15} | BGR{bgr}")

print()
print("=" * 80)
print(f"ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
print("=" * 80)
print()

# Show confusion matrix for failed cases
failed = [r for r in results if not r[4]]
if failed:
    print("FAILED CASES:")
    print("-" * 80)
    for desc, bgr, expected, detected, _ in failed:
        # Convert to HSV to show why it failed
        hsv = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        print(f"  {desc:20} | Expected: {expected:15} | Got: {detected:15}")
        print(f"  {'':20} | BGR{bgr} | HSV{tuple(hsv)}")
    print()

# Test on real webcam image
print("=" * 80)
print("REAL-WORLD TEST (Webcam)")
print("=" * 80)

try:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        h, w = frame.shape[:2]

        # Test multiple regions
        regions = [
            ("Top-Left", frame[0:h//3, 0:w//3]),
            ("Top-Center", frame[0:h//3, w//3:2*w//3]),
            ("Top-Right", frame[0:h//3, 2*w//3:w]),
            ("Center", frame[h//3:2*h//3, w//3:2*w//3]),
        ]

        print("Detected colors in different regions:")
        for name, roi in regions:
            if roi.size > 0:
                color = detector.get_dominant_color(roi)
                # Calculate average BGR
                avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
                print(f"  {name:15} | Color: {color:15} | Avg BGR{tuple(avg_bgr)}")
    else:
        print("No webcam available")
except Exception as e:
    print(f"Webcam test failed: {e}")

print()
print("=" * 80)
print("CONCLUSION:")
print("-" * 80)
print(f"Accuracy on synthetic test data: {100*correct/total:.1f}%")
print("Real-world performance: Depends on lighting and clothing variation")
print()
print("NOTES:")
print("- Synthetic tests use pure colors with added noise")
print("- Real clothing often has mixed colors, patterns, and shadows")
print("- Expected real-world accuracy: 70-85% (typical for vision systems)")
print("- Median method is resistant to shadows and highlights")
print("=" * 80)
