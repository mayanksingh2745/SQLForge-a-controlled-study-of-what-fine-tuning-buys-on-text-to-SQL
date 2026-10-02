# SQLForge Error Taxonomy & Qualitative Analysis Protocol

This document establishes the 10-category error classification taxonomy, attribution priority rules, and qualitative reporting standards for analyzing text-to-SQL generation failures.

---

## 1. The 10-Category Error Taxonomy

When a generated SQL query fails execution or returns a result set divergent from the Gold SQL query, it is assigned a primary category and optional secondary categories according to this standardized taxonomy:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SQLForge Error Taxonomy                         │
├──────┬───────────────────────────────┬─────────────────────────────────┤
│ Code │ Category                      │ Failure Description             │
├──────┼───────────────────────────────┼─────────────────────────────────┤
│ E01  │ SQL Syntax Error              │ Query fails parsing by SQLite   │
│ E02  │ Schema Hallucination / Target │ Non-existent table/column names │
│ E03  │ Join & Topology Error         │ Missing or incorrect JOIN / ON  │
│ E04  │ Aggregation & Grouping Error  │ COUNT/SUM/AVG, missing GROUP BY │
│ E05  │ Predicate & Filtering Error   │ Wrong WHERE condition / literal │
│ E06  │ Subquery & Set Error          │ Malformed IN/EXISTS/UNION/CTE   │
│ E07  │ Order, Limit & Distinct Error │ Missing ORDER BY, LIMIT, DISTINCT│
│ E08  │ Dialect Incompatibility       │ Non-SQLite syntax (ILIKE, CONCAT│
│ E09  │ Timeout & Resource Limit      │ Execution exceeded 10.0s quota  │
│ E10  │ Semantic Divergence (Silent)  │ Valid execution, wrong rows     │
│ E99  │ Unclassified Error            │ Edge cases not matching above   │
└──────┴───────────────────────────────┴─────────────────────────────────┘
```

---

## 2. Category Definitions & Concrete Examples

### E01: SQL Syntax Error
* **Definition:** The query cannot be parsed into a valid Abstract Syntax Tree (AST) by SQLite.
* **Typical Triggers:** Unmatched parentheses, missing commas between SELECT expressions, trailing operators, dangling keywords.
* **Example:**
  ```sql
  -- Model Output (E01)
  SELECT name, age FROM users WHERE age > AND active = 1;
  ```

### E02: Schema Hallucination / Selection Error
* **Definition:** The query references a table or column name that does not exist in the target schema, or selects an irrelevant column that violates intent.
* **Typical Triggers:** Inventing intuitive column names (e.g. `customer_id` instead of `cust_no`), referencing tables from another domain.
* **Example:**
  ```sql
  -- Target table has column `flight_num`; model hallucinates `flight_id`
  SELECT flight_id FROM flights WHERE origin = 'SFO';
  ```

### E03: Join & Relational Topology Error
* **Definition:** The query fails to connect necessary relational tables, joins against the wrong foreign key, or produces an unintended Cartesian cross-product.
* **Typical Triggers:** Joining `orders` to `customers` on `orders.id = customers.id` instead of `orders.customer_id = customers.id`.
* **Example:**
  ```sql
  SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.id;
  ```

### E04: Aggregation & Grouping Error
* **Definition:** Incorrect aggregation function choice (e.g. `COUNT(*)` vs `SUM(val)`), missing `GROUP BY` clause, or applying filtering in `WHERE` that should occur in `HAVING`.
* **Example:**
  ```sql
  -- Gold: SELECT dept, AVG(salary) FROM emp GROUP BY dept;
  -- Model: Missing GROUP BY
  SELECT dept, AVG(salary) FROM emp;
  ```

### E05: Predicate & Filtering Error
* **Definition:** Filtering logic is incorrect: inverted comparison operator (`<` instead of `>`), mismatched string literal case, or incorrect boolean composition (`AND` instead of `OR`).
* **Example:**
  ```sql
  -- Intent: users over 65 or students
  -- Model uses AND
  SELECT * FROM passengers WHERE age > 65 AND is_student = 1;
  ```

### E06: Subquery & Set Operation Error
* **Definition:** Syntactically valid or invalid nested subqueries that produce wrong scalar values, uncorrelated subquery errors, or incorrect set operations (`UNION` vs `INTERSECT`).
* **Example:**
  ```sql
  SELECT name FROM students WHERE gpa = (SELECT gpa FROM students); -- Subquery returns multiple rows
  ```

### E07: Order, Limit & Distinct Error
* **Definition:** Query fails to sort when ranking is required, orders ascending instead of descending, misses `LIMIT 1` for superlative questions ("Who is the oldest?"), or omits required `DISTINCT`.
* **Example:**
  ```sql
  -- Intent: Highest paid employee
  SELECT name FROM employees ORDER BY salary ASC LIMIT 1;
  ```

### E08: Dialect & Engine Incompatibility
* **Definition:** Using PostgreSQL, MySQL, or SQL Server specific functions that do not exist or behave differently in SQLite 3.
* **Typical Triggers:** Using `ILIKE`, `CONCAT(a, b)`, `DATE_ADD()`, `TOP 1`, or `TO_DATE()`.
* **Example:**
  ```sql
  SELECT * FROM users WHERE name ILIKE '%smith%'; -- ILIKE is invalid in SQLite
  ```

### E09: Timeout & Resource Limit Exceeded
* **Definition:** Query execution exceeds the 10.0-second hard wall-clock timeout or generates excessive row memory requiring process termination.
* **Typical Triggers:** Unbounded Cartesian joins (`FROM tableA, tableB, tableC` with millions of intermediate rows), recursive CTE without termination condition.

### E10: Semantic Divergence (Silent Logical Error)
* **Definition:** The query executes completely without errors and returns tabular output, but the result set does not match the Gold SQL answer.
* **Typical Triggers:** Subtle misinterpretation of domain questions, selecting the wrong metric column, or inverted sorting tied with multiple rows.

---

## 3. Assignment Protocol & Precedence Rules

Because a single query can suffer from multiple defects (e.g. hallucinating a column AND missing a GROUP BY), SQLForge assigns categories according to a strict deterministic hierarchy:

```text
1. Engine Runtime Failure?
   ├── Query timeout? ────────────────────────► E09 (Timeout)
   ├── Dialect keyword failure? ──────────────► E08 (Dialect Incompatible)
   ├── Parse error / syntax failure? ─────────► E01 (Syntax Error)
   └── Unknown table/column error? ───────────► E02 (Schema Hallucination)

2. Valid Engine Execution (Result Set Divergence)?
   ├── Cartesian join / join key mismatch? ───► E03 (Join Topology)
   ├── Aggregate / Group By mismatch? ────────► E04 (Aggregation / Grouping)
   ├── Filter / literal logic mismatch? ──────► E05 (Predicate / Filtering)
   ├── Nested query / subquery structure? ────► E06 (Subquery / Set)
   ├── Ordering / Limit / Distinct? ──────────► E07 (Order / Limit)
   └── Semantic divergence / other logic? ────► E10 (Semantic Divergence)
```

* **Primary Cause:** Assigned based on the highest-priority failing tier in the hierarchy.
* **Secondary Causes:** Up to two additional tags may be attached in `generations.jsonl` under `error_secondary_tags`.

---

## 4. Privacy & Data Protection Safeguards

To prevent leaking sensitive database contents into error analysis reports:
* **No Raw Data Values:** Error analysis logs and tables report table names, column names, error category codes, and sanitized SQL strings.
* **Literal Scrubbing:** Numerical IDs, personal phone numbers, or proprietary literal values extracted from test databases are redacted or masked (`[REDACTED_LITERAL]`) before inclusion in markdown reports or research publications.
