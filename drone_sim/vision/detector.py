"""Wire-connection and battery-cable plug detector.

Works on synthetic overhead images produced by
drone_sim/vision/vision_scenarios.py.  No deep-learning weights are needed; the
detector uses structured-image analysis (edge density, colour channels,
blob geometry) so it runs with only NumPy and SciPy.

Typical usage
-------------
    from drone_sim.vision.detector import WireConnectionDetector
    detector = WireConnectionDetector()
    result   = detector.detect(image_array)   # uint8 RGB (H, W, 3)

Detection targets
-----------------
* Wire segments        – thin, elongated dark or coloured strands
* Crimp / bullet plugs – cylindrical bright blobs at wire ends
* XT30 / XT60 plugs    – rectangular bright blobs (battery connectors)
* Battery cable pairs  – two parallel wires joined at a plug body
* Exposed solder joints / pin headers – small dense clusters

Result dictionary keys
----------------------
connections      list[dict]   Each found connection point with
                              keys: type, confidence, centroid_px,
                              bbox_px, area_px2
wire_count       int          Total distinct wire segments found
plug_count       int          Total plug bodies found
overall_score    float        0–1; higher = more confident scene
warnings         list[str]    Any quality or anomaly notes
"""

import numpy as np


# ---------------------------------------------------------------------------
# Low-level image utilities
# ---------------------------------------------------------------------------

def _to_float(image):
    """Return a float64 image in [0, 1] regardless of input dtype."""
    img = np.asarray(image, dtype = float)
    if img.ndim != 3 or img.shape[2] != 3 or not img.size or not np.all(np.isfinite(img)):
        raise ValueError("image must be a nonempty finite RGB array with shape (H, W, 3).")
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _grayscale(image_f):
    """Luminance-weighted grayscale from float RGB (H, W, 3)."""
    return 0.299 * image_f[..., 0] + 0.587 * image_f[..., 1] + 0.114 * image_f[..., 2]


def _sobel(gray):
    """Simple 3×3 Sobel edge magnitude; avoids scipy for portability."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = float)
    ky = kx.T

    def conv2d(img, k):
        """2-D correlation via stride tricks (small kernel, no scipy needed)."""
        h, w = img.shape
        kh, kw = k.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode = "edge")
        out = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * padded[i:i + h, j:j + w]
        return out

    gx = conv2d(gray, kx)
    gy = conv2d(gray, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


def _threshold_otsu(values):
    """Otsu's binarization threshold for a flat float array in [0, 1]."""
    hist, bin_edges = np.histogram(values, bins = 256, range = (0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5
    w0 = np.cumsum(hist)
    w1 = total - w0
    sum_total = float(np.sum(bin_centers * hist))
    sum0 = np.cumsum(bin_centers * hist)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        mu0 = np.where(w0 > 0, sum0 / w0, 0.0)
        mu1 = np.where(w1 > 0, (sum_total - sum0) / w1, 0.0)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(sigma_b))
    return float(bin_centers[idx])


def _label_connected(binary):
    """Simple 4-connected labelling without scipy; returns label array and count."""
    labels = np.zeros_like(binary, dtype = np.int32)
    current = 1
    h, w = binary.shape
    for r in range(h):
        for c in range(w):
            if binary[r, c] and labels[r, c] == 0:
                # Flood fill with a stack.
                stack = [(r, c)]
                labels[r, c] = current
                while stack:
                    pr, pc = stack.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            stack.append((nr, nc))
                current += 1
    return labels, current - 1


def _region_props(labels, n_labels):
    """Return list of dicts with area, centroid, bbox for each region."""
    props = []
    for lab in range(1, n_labels + 1):
        mask = labels == lab
        area = int(mask.sum())
        if area == 0:
            continue
        rows, cols = np.where(mask)
        cy = float(rows.mean())
        cx = float(cols.mean())
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        width  = c1 - c0 + 1
        height = r1 - r0 + 1
        aspect = max(width, height) / max(min(width, height), 1)
        props.append({
            "label": lab, "area": area, "centroid": (cy, cx),
            "bbox": (r0, c0, r1, c1),
            "aspect": aspect, "width": width, "height": height,
        })
    return props


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _wire_mask(image_f, edge_map, min_edge = 0.08):
    """Binary mask of thin, elongated structures (candidate wire pixels)."""
    threshold = max(_threshold_otsu(edge_map.ravel()), min_edge)
    return edge_map > threshold


def _plug_candidates(image_f, min_brightness = 0.55, min_area = 4):
    """
    Bright-blob candidates for crimp plugs, XT30/XT60 bodies, etc.
    Battery connectors tend to be shiny metal – high intensity in all channels.
    """
    gray = _grayscale(image_f)
    bright = gray > min_brightness
    labels, n = _label_connected(bright)
    props = _region_props(labels, n)
    return [p for p in props if p["area"] >= min_area]


def _colour_wire_channels(image_f):
    """
    Return per-channel binary masks for red, black, yellow, blue wires –
    common battery-cable colour coding.
    """
    r, g, b = image_f[..., 0], image_f[..., 1], image_f[..., 2]
    channels = {
        "red":    (r > 0.55) & (g < 0.35) & (b < 0.35),
        "black":  (r < 0.25) & (g < 0.25) & (b < 0.25),
        "yellow": (r > 0.60) & (g > 0.55) & (b < 0.30),
        "blue":   (b > 0.55) & (r < 0.35) & (g < 0.35),
    }
    return channels


# ---------------------------------------------------------------------------
# Connection-point classification
# ---------------------------------------------------------------------------

def _classify_plug(prop):
    """
    Heuristic: XT60 bodies are roughly square; bullet/crimp plugs are elongated.
    Returns 'xt_connector', 'crimp_plug', or 'solder_joint'.
    """
    aspect = prop["aspect"]
    area   = prop["area"]
    if aspect < 1.6 and area >= 60:
        return "xt_connector"       # roughly square, large – XT30/XT60
    if aspect >= 2.0 and area >= 12:
        return "crimp_plug"         # elongated barrel / bullet
    return "solder_joint"           # small dense blob


def _confidence_for_plug(prop, wire_density_near):
    """
    Simple confidence score: larger area + nearby wire density → higher score.
    Returns float in [0, 1].
    """
    area_score   = min(prop["area"] / 200.0, 1.0)
    wire_score   = min(wire_density_near * 5.0, 1.0)
    return float(np.clip((area_score + wire_score) / 2.0, 0.0, 1.0))


def _wire_density_near(wire_mask, centroid, radius = 12):
    """
    Fraction of pixels in a square neighbourhood that are wire-edge pixels.
    """
    h, w = wire_mask.shape
    r, c = int(round(centroid[0])), int(round(centroid[1]))
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)
    patch = wire_mask[r0:r1, c0:c1]
    total = patch.size
    return float(patch.sum() / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Wire-segment analysis
# ---------------------------------------------------------------------------

def _analyse_wires(wire_mask, colour_masks):
    """
    Return list of wire-segment dicts: colour, length estimate, bbox, centroid.
    Each colour channel is labelled independently; unlabelled edge blobs are
    reported as 'unknown'.
    """
    segments = []
    covered  = np.zeros_like(wire_mask, dtype = bool)

    for colour, cmask in colour_masks.items():
        combined = wire_mask & cmask
        if not combined.any():
            continue
        labels, n = _label_connected(combined)
        for prop in _region_props(labels, n):
            if prop["area"] < 3:
                continue
            # Length estimate: diagonal of bounding box for elongated blobs.
            diag = float(np.hypot(prop["height"], prop["width"]))
            segments.append({
                "colour": colour,
                "length_px": round(diag, 1),
                "centroid_px": (round(prop["centroid"][0], 1), round(prop["centroid"][1], 1)),
                "bbox_px": prop["bbox"],
                "area_px2": prop["area"],
            })
            # Mark as covered so we don't double-count.
            r0, c0, r1, c1 = prop["bbox"]
            covered[r0:r1 + 1, c0:c1 + 1] |= (labels[r0:r1 + 1, c0:c1 + 1] ==
                                                prop["label"])

    # Remaining uncovered wire pixels → unknown colour.
    remainder = wire_mask & (~covered)
    if remainder.any():
        labels, n = _label_connected(remainder)
        for prop in _region_props(labels, n):
            if prop["area"] < 3:
                continue
            diag = float(np.hypot(prop["height"], prop["width"]))
            segments.append({
                "colour": "unknown",
                "length_px": round(diag, 1),
                "centroid_px": (round(prop["centroid"][0], 1), round(prop["centroid"][1], 1)),
                "bbox_px": prop["bbox"],
                "area_px2": prop["area"],
            })

    return segments


def _label_for(labels, centroid):
    r, c = int(round(centroid[0])), int(round(centroid[1]))
    r = int(np.clip(r, 0, labels.shape[0] - 1))
    c = int(np.clip(c, 0, labels.shape[1] - 1))
    return int(labels[r, c])


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def _check_warnings(image_f, wire_mask, connections):
    """Return list of human-readable warning strings."""
    warnings = []
    h, w = image_f.shape[:2]

    # Very dark image – poor lighting.
    mean_brightness = float(_grayscale(image_f).mean())
    if mean_brightness < 0.10:
        warnings.append("Image is very dark; detection confidence may be low.")

    # Very blurry – edge density extremely low.
    edge_fraction = float(wire_mask.mean())
    if edge_fraction < 0.005:
        warnings.append("Few edge features detected; image may be out of focus.")

    # Possible bare / exposed wire (no plug at end of a wire segment).
    if connections:
        plug_types = {c["type"] for c in connections}
        if "solder_joint" in plug_types and "xt_connector" not in plug_types and "crimp_plug" not in plug_types:
            warnings.append("Only solder joints found; verify battery cable terminations.")

    # Image resolution very small.
    if h < 32 or w < 32:
        warnings.append(f"Image is small ({w}×{h} px); accuracy may be reduced.")

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class WireConnectionDetector:
    """Detect wire connections and battery-cable plugs in overhead drone images.

    Parameters
    ----------
    min_plug_area : int
        Minimum blob area (pixels²) to be considered a plug candidate.
    min_plug_brightness : float
        Minimum mean brightness [0–1] for a candidate plug region.
    wire_neighbourhood_radius : int
        Pixel radius used when computing wire density around a plug.
    """

    def __init__(self, min_plug_area = 8, min_plug_brightness = 0.50,
                 wire_neighbourhood_radius = 12):
        self.min_plug_area              = int(min_plug_area)
        self.min_plug_brightness        = float(min_plug_brightness)
        self.wire_neighbourhood_radius  = int(wire_neighbourhood_radius)
        if self.min_plug_area < 1 or self.wire_neighbourhood_radius < 0:
            raise ValueError("Plug area must be positive and neighbourhood radius nonnegative.")
        if not np.isfinite(self.min_plug_brightness) or not 0 <= self.min_plug_brightness <= 1:
            raise ValueError("Plug brightness must be between 0 and 1.")

    # ------------------------------------------------------------------
    def detect(self, image):
        """Run the full detection pipeline on one image.

        Parameters
        ----------
        image : array-like, shape (H, W, 3), uint8 or float
            RGB image.  Values may be 0–255 (uint8) or 0–1 (float).

        Returns
        -------
        dict with keys:
            connections     list[dict]
            wire_count      int
            plug_count      int
            overall_score   float
            warnings        list[str]
        """
        image_f = _to_float(image)
        if image_f.ndim != 3 or image_f.shape[2] != 3:
            raise ValueError("image must be an RGB array with shape (H, W, 3).")

        gray        = _grayscale(image_f)
        edge_map    = _sobel(gray)
        # Normalise edge map to [0, 1] for threshold consistency.
        emax = edge_map.max()
        edge_norm   = edge_map / emax if emax > 0 else edge_map

        wire_mask     = _wire_mask(image_f, edge_norm)
        colour_masks  = _colour_wire_channels(image_f)
        plug_props    = _plug_candidates(image_f,
                                         min_brightness = self.min_plug_brightness,
                                         min_area       = self.min_plug_area)
        wire_segments = _analyse_wires(wire_mask, colour_masks)

        connections = []
        for prop in plug_props:
            density = _wire_density_near(wire_mask, prop["centroid"],
                                         radius = self.wire_neighbourhood_radius)
            ptype  = _classify_plug(prop)
            conf   = _confidence_for_plug(prop, density)
            connections.append({
                "type":        ptype,
                "confidence":  round(conf, 3),
                "centroid_px": (round(prop["centroid"][0], 1),
                                round(prop["centroid"][1], 1)),
                "bbox_px":     prop["bbox"],
                "area_px2":    prop["area"],
            })

        # Sort by confidence descending.
        connections.sort(key = lambda c: c["confidence"], reverse = True)

        wire_count  = len(wire_segments)
        plug_count  = len(connections)
        score       = self._overall_score(connections, wire_count)
        warns       = _check_warnings(image_f, wire_mask, connections)

        return {
            "connections":   connections,
            "wire_segments": wire_segments,
            "wire_count":    wire_count,
            "plug_count":    plug_count,
            "overall_score": round(score, 4),
            "warnings":      warns,
        }

    def detect_bgr(self, frame):
        """Camera adapter: convert OpenCV BGR to the detector's RGB contract."""
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3).")
        return self.detect(image[..., ::-1])

    # ------------------------------------------------------------------
    def _overall_score(self, connections, wire_count):
        """
        Aggregate confidence signal.  Rewards finding at least one plug and
        some wire segments.  Penalises zero wires (probably no cable present).
        """
        if not connections:
            return 0.0
        mean_conf  = float(np.mean([c["confidence"] for c in connections]))
        wire_bonus = min(wire_count / 5.0, 1.0) * 0.20
        return float(np.clip(mean_conf * 0.80 + wire_bonus, 0.0, 1.0))
