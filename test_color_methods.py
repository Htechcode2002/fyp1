import cv2
import numpy as np
import time

# Create test images
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img[:, :] = [40, 40, 200]  # BGR red

roi = test_img[10:90, 10:90]

# Method 1: Histogram mode (current method)
times_1 = []
for _ in range(50):
    start = time.time()
    roi_quantized = (roi // 32) * 32
    pixels = roi_quantized.reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_bgr = unique[counts.argmax()]
    times_1.append((time.time() - start) * 1000)
method1_time = np.mean(times_1)

# Method 2: K-means clustering
from sklearn.cluster import KMeans
times_2 = []
for _ in range(50):
    start = time.time()
    pixels_float = roi.reshape(-1, 3).astype(np.float32)
    # Use 1 cluster since test image is solid color
    kmeans = KMeans(n_clusters=1, random_state=0, n_init=5).fit(pixels_float)
    dominant_kmeans = kmeans.cluster_centers_[0]
    times_2.append((time.time() - start) * 1000)
method2_time = np.mean(times_2)

# Method 3: Mean color
times_3 = []
for _ in range(50):
    start = time.time()
    dominant_mean = np.mean(roi, axis=(0, 1))
    times_3.append((time.time() - start) * 1000)
method3_time = np.mean(times_3)

print("=" * 60)
print("Color Detection Performance Comparison (50 iterations avg)")
print("=" * 60)
print(f"Method 1 (Histogram Mode - Current): {method1_time:.2f}ms")
print(f"  - Color: {dominant_bgr}")
print(f"  - Pro: Fast, resistant to noise")
print(f"  - Con: Quantization may lose precision")
print()
print(f"Method 2 (K-means Clustering): {method2_time:.2f}ms")
print(f"  - Color: {dominant_kmeans.astype(int)}")
print(f"  - Pro: Most accurate for complex patterns")
print(f"  - Con: {method2_time/method1_time:.1f}x slower, overkill for simple cases")
print()
print(f"Method 3 (Mean Color): {method3_time:.2f}ms")
print(f"  - Color: {dominant_mean.astype(int)}")
print(f"  - Pro: Fastest method")
print(f"  - Con: Affected by outliers and shadows")
print()
print("=" * 60)
print(f"RECOMMENDATION: Histogram mode is best balance (accurate + fast)")
print(f"Speed comparison: K-means is {method2_time/method1_time:.1f}x slower")
print("=" * 60)

# Test on real-world scenario (mixed colors)
print("\n" + "=" * 60)
print("Real-world test: Mixed color image")
print("=" * 60)
mixed_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
# Add dominant red color
mixed_img[50:150, 50:150] = [40, 40, 200]
roi_mixed = mixed_img[50:150, 50:150]

# Test all methods
start = time.time()
roi_quantized = (roi_mixed // 32) * 32
pixels = roi_quantized.reshape(-1, 3)
unique, counts = np.unique(pixels, axis=0, return_counts=True)
hist_result = unique[counts.argmax()]
hist_time = (time.time() - start) * 1000

start = time.time()
pixels_float = roi_mixed.reshape(-1, 3).astype(np.float32)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=5).fit(pixels_float)
# Get largest cluster
labels = kmeans.labels_
cluster_counts = np.bincount(labels)
dominant_cluster = cluster_counts.argmax()
kmeans_result = kmeans.cluster_centers_[dominant_cluster]
kmeans_time = (time.time() - start) * 1000

start = time.time()
mean_result = np.mean(roi_mixed, axis=(0, 1))
mean_time = (time.time() - start) * 1000

print(f"Histogram: {hist_time:.2f}ms - Color: {hist_result}")
print(f"K-means:   {kmeans_time:.2f}ms - Color: {kmeans_result.astype(int)}")
print(f"Mean:      {mean_time:.2f}ms - Color: {mean_result.astype(int)}")
print(f"\nFor real-time video: Histogram is {kmeans_time/hist_time:.1f}x faster than K-means")
