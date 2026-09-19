# Feature Schema Verification & Column Mapping Report

**File Verified:** `elliptic_txs_features.csv`  
**Verification Method:** Programmatic headerless parsing & boundary verification  
**Total Raw CSV Columns:** **167 columns** (Indices `0` through `166`)  

---

## 1. Authoritative Column Partition Summary

| Partition Category | Column Indices | Column Count | Machine Learning Role |
| :--- | :--- | :--- | :--- |
| **Transaction ID** | `[0]` | 1 column | Primary Key / Index (`txId`) — **Excluded from features** |
| **Temporal Epoch** | `[1]` | 1 column | Splitting criterion (`time_step` in [1, 49]) — **Excluded from features** |
| **Local Features** | `[2, 94]` | **93 features** | Local transaction attributes (Configuration A & B) |
| **Aggregated Features** | `[95, 166]` | **72 features** | 1-hop neighborhood statistics (Configuration B only) |
| **Total ML Features** | `[2, 166]` | **165 features** | **Authoritative tabular feature space** |

Total Columns (167) = 1 (txId) + 1 (time_step) + 93 (Local Features) + 72 (Aggregated Features)

---

## 2. Mathematical Reconciliation of "166 vs 165 Features"

1. **Non-ID Columns Count:** Columns 1 to 166 = 166 columns (1 time_step + 165 ML Features).
2. **0-Indexed Maximum Index:** The highest column index in Python is `166`.
3. **Academic Standard (Weber et al., KDD 2019):** Exactly **93 local + 72 aggregated = 165 ML features**.

This confirms that the feature set used in Phase 2 experiments strictly adheres to the 165-feature standard.
