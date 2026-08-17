# Data Module — DataCo Dataset Decisions

Findings from `notebooks/01_eda.ipynb` that directly affect preprocessing and model code.

## Data Loading

- Encoding: `pd.read_csv(..., encoding='latin-1')` — file contains accented characters.
- Strip column names on load: `df.columns = df.columns.str.strip()`.
- Parse date columns with `pd.to_datetime(..., format='%m/%d/%Y %H:%M')`.
- Round `Order Item Discount Rate` to 2 decimal places to remove float noise.

## Confirmed Leakage — Never use these as ML features

| Column | Reason |
|---|---|
| `Delivery Status` | Direct text encoding of the target (`Late_delivery_risk`) |
| `Days for shipping (real)` | Actual shipping days — only observable post-delivery |
| `shipping date (DateOrders)` | Actual ship date — post-fulfillment |
| `Order Status` | Fulfillment status (COMPLETE / CANCELED / etc.) finalized after delivery |
| `Benefit per order` | Post-fulfillment profit |
| `Order Profit Per Order` | Post-fulfillment profit |
| `Order Item Profit Ratio` | Post-fulfillment profit ratio |
| `Sales per customer` | Aggregate across orders — cross-order leakage |

## Columns to Drop

- **PII / masked**: `Customer Email`, `Customer Password` (all `XXXXXXXXX`)
- **Redundant IDs**: `Customer Id` (identical to `Order Customer Id`), `Order Customer Id`, `Order Id`, `Order Item Id`, `Order Item Cardprod Id` (identical to `Product Card Id`), `Product Card Id`
- **Redundant categoricals**: `Category Id` (use `Category Name`), `Department Id` (use `Department Name`), `Product Category Id` (duplicate of `Category Id`)
- **Useless**: `Product Description` (100% null), `Product Image` (URLs), `Product Status` (constant 0), `Customer Email`, `Customer Fname`, `Customer Lname`
- **Customer location** (store registration address, not delivery destination): `Customer City`, `Customer Country`, `Customer State`, `Customer Street`, `Customer Zipcode`
- **Missing**: `Order Zipcode` (>95% missing)
- **Duplicates**: `Order Item Product Price` (same as `Product Price`), `Order Item Discount` (use rate instead), `Sales` (alias for `Order Item Total`)
- **High cardinality with no gain**: `Product Name` (Category/Department already cover it)
- **Redundant numeric shadow**: `Days for shipment (scheduled)` — perfectly maps 1-to-1 with `Shipping Mode` (Same Day=0, First Class=1-2, Standard Class=4, etc.). The mode is the causal variable and carries richer semantic meaning (service level, handling priority); the day count is just a numeric side-effect. Drop this and keep `Shipping Mode`.

## Approved ML Feature Set (Phase 2)

| Feature | Type |
|---|---|
| `Shipping Mode` | categorical |
| `Market` | categorical |
| `Order Region` | categorical |
| `Order Country` | categorical |
| `Customer Segment` | categorical |
| `Category Name` | categorical |
| `Department Name` | categorical |
| `Type` | categorical |
| `Order Item Quantity` | numeric |
| `Order Item Total` | numeric |
| `Product Price` | numeric |
| `Order Item Discount Rate` | numeric |
| `order_month` | numeric (derived from `order date (DateOrders)`) |
| `order_quarter` | numeric (derived from `order date (DateOrders)`) |
| `route_historical_delay_rate` | numeric (derived, leakage-safe — see below) |

Target: `Late_delivery_risk` (binary, 55% positive — mild imbalance, no resampling needed).

### `route_historical_delay_rate` — construction and leakage safety

Lane = `(Market, Order Region)`. Deliberately **not** tied to the Phase 1/3 k-means
facility clusters — this feature describes the shipment (market + destination region),
not which candidate facility might serve it, so it stays valid regardless of how Phase 1
re-clusters facilities. One consequence worth remembering going into Phase 3: since no
feature encodes facility identity, `P(delay_ij)` predictions currently vary by
destination region but not by which facility `i` serves it — risk-adjustment differentiates
by *lane destination*, not by *facility choice*, for a given region.

Computed in `preprocessor.add_route_historical_delay_rate`, called from `clean()` before
`order date (DateOrders)` is dropped:

1. Sort all rows by `order date (DateOrders)`.
2. For each row, take only shipments on the same lane that occurred **strictly earlier**
   in time — never the row's own label, never a future shipment.
3. Bayesian-smooth the lane's raw prior rate toward a dataset-wide rate, weighted by
   `route_history_smoothing_k` (`config.py`, default 20) vs. how many prior shipments the
   lane has — new/thin lanes lean on the global rate, established lanes trust their own
   history.
4. The smoothing anchor (dataset-wide rate) is *also* point-in-time (expanding, prior-only)
   rather than a static full-dataset mean, so it can't leak future information either, just
   more diffusely than the per-lane version would.

This is safe under the current random train/test split because the leakage rule is
per-row-in-time, not per-split: each row only ever looks at its own strictly-earlier
history regardless of which partition it lands in.

## CFLP Inputs (Phases 1 & 3)

- **Facility nodes**: unique `(Latitude, Longitude)` pairs from the dataset — cluster to a tractable `k` before solving.
- **Demand nodes**: `Order City` / `Order Country` — requires geocoding, or use `Order Region` (22 groups) to avoid it.
- **Demand volume** `d_j`: aggregate `Order Item Total` per demand node.
- **Route cost** `c_ij`: haversine distance between facility and demand node coordinates.
