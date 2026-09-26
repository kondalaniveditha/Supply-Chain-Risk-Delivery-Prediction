# DATA_DICTIONARY.md

## Supply Chain Risk Analytics — DataCo Dataset

**Source:** DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS  
**File:** `data/raw/DataCoSupplyChainDataset.csv`  
**Rows:** 180,519 | **Columns:** 53  
**Encoding:** Latin-1

---

## Column Roles Legend

| Role | Meaning |
|------|---------|
| **Feature** | Used as input to the prediction model |
| **Target** | The variable being predicted |
| **Leakage** | Post-delivery information — excluded from features |
| **PII** | Personally identifiable information — excluded entirely |
| **Identifier** | Meaningless unique ID — excluded |
| **Excluded** | Sparse, redundant, or low-signal — excluded |

---

## Full Data Dictionary

| Column | Meaning | Data Type | Example | Role | Reason |
|--------|---------|-----------|---------|------|--------|
| Type | Payment type (DEBIT, TRANSFER, CASH, PAYMENT) | Categorical | DEBIT | **Feature** | Pre-delivery transaction info |
| Days for shipping (real) | Actual number of days taken to ship | Integer | 3 | **Target (Regression)** | What we predict |
| Days for shipment (scheduled) | Number of scheduled delivery days | Integer | 4 | **Feature** | Pre-delivery schedule commitment |
| Benefit per order | Profit earned per order | Float | 91.25 | **Feature** | Order profitability signal |
| Sales per customer | Total sales for the customer | Float | 314.64 | **Feature** | Customer value signal |
| Delivery Status | Text description of delivery outcome | Categorical | Late delivery | **Leakage** | Only known post-delivery; derived from Late_delivery_risk |
| Late_delivery_risk | Binary flag: 1=late, 0=on time | Integer (0/1) | 0 | **Target (Classification)** | What we predict |
| Category Id | Numeric product category code | Integer | 73 | **Identifier** | Redundant with Category Name |
| Category Name | Text name of product category | Categorical | Sporting Goods | **Feature** | Category influences logistics |
| Customer City | City of customer | Categorical | Caguas | **Excluded** | High cardinality, PII-adjacent |
| Customer Country | Country of customer | Categorical | Puerto Rico | **Excluded** | Redundant with Market |
| Customer Email | Customer email address | String | XXXXXXXXX | **PII** | Personally identifiable |
| Customer Fname | Customer first name | String | Cally | **PII** | Personally identifiable |
| Customer Id | Unique customer ID number | Integer | 20755 | **Identifier** | Meaningless for prediction |
| Customer Lname | Customer last name | String | Holloway | **PII** | Personally identifiable |
| Customer Password | Masked password field | String | XXXXXXXXX | **PII** | Security sensitive |
| Customer Segment | Business segment (Consumer/Corporate/Home Office) | Categorical | Consumer | **Feature** | Different segments may have different patterns |
| Customer State | US state of customer | Categorical | PR | **Excluded** | PII-adjacent, redundant with Market |
| Customer Street | Street address of customer | String | 5365 Noble Nectar Island | **PII** | Personally identifiable |
| Customer Zipcode | Postal code of customer | Float | 725.0 | **PII** | Personally identifiable, sparse |
| Department Id | Numeric store department ID | Integer | 2 | **Identifier** | Redundant with Department Name |
| Department Name | Text name of the department | Categorical | Fitness | **Feature** | Department logistics differ |
| Latitude | Latitude of store location | Float | 18.25 | **Excluded** | Noisy geo proxy, high collinearity |
| Longitude | Longitude of store location | Float | -66.04 | **Excluded** | Noisy geo proxy, high collinearity |
| Market | Geographic market (Africa/Europe/LATAM/Pacific Asia/USCA) | Categorical | Pacific Asia | **Feature** | Market drives shipping distance and infrastructure |
| Order City | City of destination | Categorical | Bekasi | **Excluded** | Very high cardinality |
| Order Country | Country of destination | Categorical | Indonesia | **Excluded** | Redundant with Market/Region |
| Order Customer Id | Customer ID linked to order | Integer | 20755 | **Identifier** | Redundant |
| order date (DateOrders) | Timestamp when order was placed | String/DateTime | 1/31/2018 22:56 | **Feature** (engineered) | Parsed into date components |
| Order Id | Unique order identifier | Integer | 77202 | **Identifier** | Meaningless for prediction |
| Order Item Cardprod Id | Product RFID card code | Integer | 1360 | **Identifier** | Internal product code |
| Order Item Discount | Dollar value of item discount applied | Float | 13.11 | **Feature** | Discount may signal rush/promotional orders |
| Order Item Discount Rate | Percentage discount rate | Float | 0.04 | **Feature** | Discount rate signal |
| Order Item Id | Unique order item identifier | Integer | 180517 | **Identifier** | Meaningless |
| Order Item Product Price | Product price before discount | Float | 327.75 | **Feature** | Price influences handling and logistics |
| Order Item Profit Ratio | Profit ratio per order item | Float | 0.29 | **Feature** | Profitability signal |
| Order Item Quantity | Number of units ordered | Integer | 1 | **Feature** | Quantity affects fulfillment time |
| Sales | Revenue from the order item | Float | 327.75 | **Feature** | Order value |
| Order Item Total | Total order item amount | Float | 314.64 | **Feature** | Total value |
| Order Profit Per Order | Profit per complete order | Float | 91.25 | **Feature** | Profitability signal |
| Order Region | Geographic sub-region of destination | Categorical | Southeast Asia | **Feature** | Regional logistics differ significantly |
| Order State | State/province within the region | Categorical | Java Occidental | **Excluded** | Very high cardinality, redundant with Order Region |
| Order Status | Order workflow status (COMPLETE/PENDING/etc.) | Categorical | COMPLETE | **Feature** | Order lifecycle stage at time of prediction |
| Order Zipcode | ZIP code of destination | Float | NaN (86% missing) | **Excluded** | ~86% missing values |
| Product Card Id | Internal product identifier | Integer | 1360 | **Identifier** | Redundant with Product Name |
| Product Category Id | Numeric product category code | Integer | 73 | **Identifier** | Redundant with Category Name |
| Product Description | Text description of product | Float | NaN (100% missing) | **Excluded** | 100% missing |
| Product Image | URL to product image | String | http://... | **Excluded** | URL, no predictive signal |
| Product Name | Name of the product | Categorical | Smart watch | **Excluded** | Very high cardinality, proxied by Category Name |
| Product Price | Retail price of product | Float | 327.75 | **Feature** | High-value items may have different logistics |
| Product Status | Stock availability (0=available, 1=unavailable) | Integer (0/1) | 0 | **Feature** | Unavailable products may introduce delays |
| shipping date (DateOrders) | Actual shipment departure timestamp | String/DateTime | 2/3/2018 22:56 | **Leakage** | Only known after shipment — post-delivery |
| Shipping Mode | Shipping method (Standard/First/Second/Same Day) | Categorical | Standard Class | **Feature** | Strongest predictor of delivery time |

---

## Engineered Features (created during preprocessing)

| Column | Description | Source |
|--------|-------------|--------|
| order_year | Year of order placement | Parsed from order date |
| order_month | Month of order placement (1–12) | Parsed from order date |
| order_quarter | Quarter of order placement (1–4) | Parsed from order date |
| order_day | Day of month of order | Parsed from order date |
| order_dayofweek | Day of week (0=Mon, 6=Sun) | Parsed from order date |
| order_hour | Hour of day order was placed | Parsed from order date |
| is_expedited_schedule | 1 if scheduled days ≤ 2 | Days for shipment (scheduled) |
| profit_margin_ratio | Order Profit / Order Item Total | Derived from order financials |
| has_discount | 1 if item discount > 0 | Order Item Discount |
| is_high_value | 1 if Product Price ≥ $200 | Product Price |
| is_weekend_order | 1 if order placed on Saturday or Sunday | order_dayofweek |

---

## Data Quality Summary

| Issue | Detail |
|-------|--------|
| Missing values | Product Description: 100% missing; Order Zipcode: 86% missing; Customer Lname: 8 missing |
| Duplicate rows | 0 exact duplicates found |
| PII columns removed | Customer Email, Fname, Lname, Password, Street, Zipcode |
| Leakage columns removed | Delivery Status, shipping date (DateOrders) |
| Identifiers removed | Customer Id, Order Id, Order Item Id, Order Item Cardprod Id, Product Card Id, Department Id, Category Id, Product Category Id, Order Customer Id |
| Sparse/low-signal removed | Product Description, Product Image, Order Zipcode, Order State, Customer State, Customer City, Order City, Order Country, Customer Country, Latitude, Longitude |

---

## Target Variable Analysis

| Target | Type | Distribution |
|--------|------|-------------|
| Days for shipping (real) | Regression | Mean=3.50 days, Std=1.62, Min=0, Max=6 |
| Late_delivery_risk | Classification | 1 (Late)=54.8%, 0 (On Time)=45.2% — mild class imbalance |

**Note on class imbalance:** The classification target has ~55% late deliveries vs ~45% on-time.
This mild imbalance is addressed using `class_weight='balanced'` in applicable models.
