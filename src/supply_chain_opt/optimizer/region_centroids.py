# Approximate geographic centroids for DataCo's 23 Order Region values.
# Keys match the raw parquet strings exactly — three have trailing/double spaces.
# See assumptions.md for why hand-specified centroids are used instead of geocoding.
REGION_CENTROIDS: dict[str, tuple[float, float]] = {
    "Canada":           (56.13, -106.35),
    "Caribbean":        (18.22,  -66.59),
    "Central Africa":   (-4.04,   21.76),
    "Central America":  (12.77,  -85.60),
    "Central Asia":     (41.20,   74.76),
    "East Africa":      (-1.29,   36.82),
    "East of USA":      (35.78,  -78.64),
    "Eastern Asia":     (35.86,  104.19),
    "Eastern Europe":   (52.00,   30.00),
    "North Africa":     (28.03,    1.66),
    "Northern Europe":  (60.13,   18.64),
    "Oceania":          (-25.27, 133.78),
    "South America":    (-14.24,  -51.93),
    "South Asia":       (20.59,   78.96),
    "South of  USA ":   (30.50,  -90.00),
    "Southeast Asia":   (12.88,  101.99),
    "Southern Africa":  (-30.56,  22.94),
    "Southern Europe":  (41.87,   12.57),
    "US Center ":       (39.50,  -98.35),
    "West Africa":      (12.35,   -1.56),
    "West Asia":        (29.31,   47.48),
    "West of USA ":     (37.09, -119.00),
    "Western Europe":   (48.69,    9.14),
}
