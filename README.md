# 🏭 Production Balancer - Advanced Manufacturing Optimization System

A sophisticated production planning and scheduling system that optimizes equipment allocation, minimizes changeovers, and maximizes throughput while respecting real-world manufacturing constraints.

## 📋 Table of Contents
- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

---

## 🎯 Overview

This system solves a complex **Multi-Product, Multi-Operation, Multi-Equipment Production Allocation Problem** using Mixed Integer Linear Programming (MILP). It considers:

- **Daily demand targets** for multiple products
- **Sequential operations** with work-in-process (WIP) flow constraints
- **Equipment capabilities** and current workload
- **Tool availability** and sharing mechanisms
- **Changeover penalties** for product/operation switches
- **Job continuation preferences** for in-progress equipment

---

## 📐 Mathematical Formulation

### Sets and Indices

| Symbol | Description |
|--------|-------------|
| $P$ | Set of products |
| $O$ | Set of operations (sequential) |
| $U$ | Set of equipment units |
| $M$ | Set of equipment models |
| $(p, o, u)$ | Valid combination of product $p$, operation $o$, and unit $u$ |

### Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| $D_p$ | Daily demand for product $p$ | units |
| $T_{p,o,m}$ | Cycle time for product $p$, operation $o$ on model $m$ | minutes/unit |
| $A$ | Available time per equipment per day | minutes (1440) |
| $W_{p,o}$ | Work-in-process available for product $p$ at operation $o$ | units |
| $E_u$ | Remaining time for current job on equipment $u$ | minutes |
| $L_{p,o}$ | Tool capacity (quantity) for product $p$, operation $o$ | count |
| $C_{prod}$ | Changeover time for product switch | minutes |
| $C_{oper}$ | Changeover time for operation switch | minutes |

### Decision Variables

| Variable | Type | Description |
|----------|------|-------------|
| $x_{p,o,u}$ | Continuous | Quantity of product $p$, operation $o$ assigned to unit $u$ |
| $y_{p,o,u}$ | Binary | 1 if unit $u$ is assigned to $(p, o)$, 0 otherwise |
| $z_{p,o}$ | Continuous | Unmet demand for product $p$ at operation $o$ |

### Objective Function

Minimize the weighted sum of penalties:

$$
\min \quad \alpha \sum_{p,o} z_{p,o} + \beta \sum_{u \in U_{wip}} \sum_{\substack{(p,o,u) \in V \\ (p,o) \neq (p_u, o_u)}} y_{p,o,u} + \gamma \sum_{(p,o,u) \in V} y_{p,o,u} + \delta \sum_{(p,o,u) \in V} x_{p,o,u}
$$

**Penalty Weights:**
- $\alpha = 1{,}000{,}000$ : Unmet demand (highest priority)
- $\beta = 10{,}000$ : Job discontinuation penalty
- $\gamma = 1{,}000$ : Assignment count (changeover proxy)
- $\delta = 1$ : Overproduction penalty

where:
- $U_{wip}$ = Set of equipment with work-in-progress
- $(p_u, o_u)$ = Current job on equipment $u$
- $V$ = Set of all valid $(p, o, u)$ combinations

---

### Constraints

#### 1. Assignment Activation Constraint
$$
x_{p,o,u} \leq M \cdot y_{p,o,u} \quad \forall (p,o,u) \in V
$$
- $M = 100{,}000$ (big-M constant)
- Ensures quantity can only be assigned if the binary assignment variable is active

#### 2. Demand Fulfillment Constraint
$$
\sum_{u \in U_{p,o_{last}}} x_{p,o_{last},u} + W_{p,o_{last}} + z_{p,o_{last}} \geq D_p \quad \forall p \in P
$$
- $o_{last}$ = Final operation in the sequence
- Production + existing WIP + unmet demand must satisfy total demand

#### 3. Flow Conservation Constraints

**For the first operation ($o_1$):**
$$
\sum_{u \in U_{p,o_1}} x_{p,o_1,u} \leq W_{p,o_1} \quad \forall p \in P
$$
- Production is limited by available input material (raw material or WIP)

**For subsequent operations ($o_i, i > 1$):**
$$
\sum_{u \in U_{p,o_i}} x_{p,o_i,u} \leq W_{p,o_i} + \sum_{u \in U_{p,o_{i-1}}} x_{p,o_{i-1},u} \quad \forall p \in P, \forall o_i \in O \setminus \{o_1\}
$$
- Each operation can only process material from (WIP + previous operation output)

#### 4. Tool Capacity Constraint (Resource Sharing)
$$
\sum_{u \in U_{p,o}} \left( x_{p,o,u} \cdot T_{p,o,m(u)} \right) \leq L_{p,o} \cdot A \quad \forall (p,o)
$$
- Total tool-hours consumed cannot exceed available tool-hours
- Enables sequential tool sharing (return and reuse)

#### 5. Equipment Time Capacity Constraint
$$
\sum_{(p,o) \in V_u} \left( x_{p,o,u} \cdot T_{p,o,m(u)} \right) \leq A - E_u \quad \forall u \in U
$$
- $V_u$ = Set of $(p,o)$ combinations valid for unit $u$
- $E_u$ = Time already occupied by current job
- Total processing time cannot exceed available time after accounting for in-progress jobs

---

## 🚀 Key Features

### 1. Multi-Environment Support
- **Production Mode**: Connects to production Oracle DB
- **Development Mode**: Uses development DB for testing
- **Local Test Mode**: Runs with sample data (no DB required)

### 2. Advanced Constraint Modeling
- **WIP Flow Control**: Strict material flow between sequential operations
- **Tool Sharing Logic**: Implements tool-hour capacity for sequential reuse
- **Job Continuation**: Prioritizes continuing in-progress jobs to minimize changeovers
- **Equipment Workload**: Accounts for current jobs and remaining time

### 3. Automated Scheduling
- **APScheduler Integration**: Periodic batch execution (configurable interval)
- **Async Job Queue**: Non-blocking API with background workers
- **Admin Dashboard**: Real-time monitoring and configuration

### 4. Visualization & Monitoring
- **Gantt Chart**: Timeline view of equipment schedules
- **Workload Analysis**: Per-unit utilization metrics
- **Unmet Demand Alerts**: Highlights infeasible scenarios

---

## 🏗️ System Architecture

```
production_balancer/
├── config/
│   ├── config.yaml          # System configuration (DB, scheduler, modes)
│   └── data_config.py       # Sample data for local testing
├── core/
│   ├── optimizer.py         # MILP model implementation (PuLP)
│   └── job_manager.py       # Async job queue & scheduler
├── database/
│   └── manager.py           # Oracle DB connector & data fetcher
├── api.py                   # FastAPI backend
├── app.py                   # Streamlit dashboard (user)
├── admin_app.py             # Streamlit admin panel
└── main.py                  # CLI verification tool
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Oracle Instant Client (for production/dev modes)

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `pulp` - Linear programming solver
- `oracledb` - Oracle database connector
- `fastapi` - REST API framework
- `streamlit` - Dashboard UI
- `apscheduler` - Batch scheduling
- `plotly` - Interactive charts

---

## 🎮 Usage

### 1. Configure System Mode
Edit `config/config.yaml`:
```yaml
system_mode: local_test  # Options: production, development, local_test

database:
  production:
    user: PROD_USER
    password: PROD_PASS
    dsn: prod-db:1521/ORCL
  development:
    user: DEV_USER
    password: DEV_PASS
    dsn: dev-db:1521/ORCL

scheduler:
  enabled: true
  interval_min: 60  # Run every 60 minutes
```

### 2. Start Backend API
```bash
python api.py
```
API will be available at `http://localhost:8000`

### 3. Launch User Dashboard
```bash
streamlit run app.py
```
Access at `http://localhost:8501`

### 4. Launch Admin Panel
```bash
streamlit run admin_app.py
```
Access at `http://localhost:8502`

### 5. CLI Verification (Optional)
```bash
python main.py
```

---

## ⚙️ Configuration

### Database Schema

The system expects the following Oracle tables:

#### TB_PRODUCTION_PLAN
| Column | Type | Description |
|--------|------|-------------|
| PRODUCT_ID | VARCHAR2 | Product identifier |
| DEMAND_QTY | NUMBER | Daily demand quantity |

#### TB_EQUIPMENT_MASTER
| Column | Type | Description |
|--------|------|-------------|
| MODEL_ID | VARCHAR2 | Equipment model |
| UNIT_ID | VARCHAR2 | Unique equipment ID |

#### TB_PROCESS_STANDARD
| Column | Type | Description |
|--------|------|-------------|
| PRODUCT_ID | VARCHAR2 | Product identifier |
| OPER_ID | VARCHAR2 | Operation identifier |
| MODEL_ID | VARCHAR2 | Compatible model |
| CYCLE_TIME | NUMBER | Seconds per unit |

#### TB_WIP_STATUS
| Column | Type | Description |
|--------|------|-------------|
| PRODUCT_ID | VARCHAR2 | Product identifier |
| OPER_ID | VARCHAR2 | Operation identifier |
| WIP_QTY | NUMBER | Available WIP quantity |

#### TB_EQP_WIP
| Column | Type | Description |
|--------|------|-------------|
| EQP_ID | VARCHAR2 | Equipment ID |
| PROD_ID | VARCHAR2 | Current product |
| OPER_ID | VARCHAR2 | Current operation |
| END_TIME | TIMESTAMP | Expected completion time |

#### TB_TOOL_MASTER
| Column | Type | Description |
|--------|------|-------------|
| PRODUCT_ID | VARCHAR2 | Product identifier |
| OPER_ID | VARCHAR2 | Operation identifier |
| TOOL_QTY | NUMBER | Available tool count |

#### TB_PRODUCTION_RESULTS (Output)
| Column | Type | Description |
|--------|------|-------------|
| UNIT_ID | VARCHAR2 | Equipment ID |
| PRODUCT_ID | VARCHAR2 | Assigned product |
| OPER_ID | VARCHAR2 | Assigned operation |
| QUANTITY | NUMBER | Allocated quantity |
| START_TIME | TIMESTAMP | Scheduled start |
| END_TIME | TIMESTAMP | Scheduled end |

---

## 📊 Sample Data Structure

```python
# Daily Demand
DEMAND = {
    'Product_A': 100,
    'Product_B': 100
}

# Equipment Models
EQUIPMENT_MODELS = {
    'Model_X': ['Unit_1', 'Unit_2'],  # OP10 capable
    'Model_Y': ['Unit_3', 'Unit_4']   # OP20 capable
}

# Cycle Times (minutes per unit)
PROCESS_CONFIG = {
    ('Product_A', 'OP10', 'Model_X'): 1.5,
    ('Product_B', 'OP10', 'Model_X'): 2.0,
    ('Product_A', 'OP20', 'Model_Y'): 2.5,
    ('Product_B', 'OP20', 'Model_Y'): 3.0,
}

# Work-in-Process (Input per Stage)
WIP = {
    ('Product_A', 'OP10'): 200,  # Raw material for OP10
    ('Product_A', 'OP20'): 0,    # No pre-existing WIP for OP20
    ('Product_B', 'OP10'): 200,
    ('Product_B', 'OP20'): 0,
}

# Equipment Current Jobs (minutes remaining)
EQP_WIP = {
    'Unit_1': {'Product': 'Product_A', 'Operation': 'OP10', 'End_Time_Offset': 10},
    'Unit_3': {'Product': 'Product_B', 'Operation': 'OP20', 'End_Time_Offset': 5},
}

# Tool Availability
TOOLS = {
    ('Product_A', 'OP10'): 1,  # Only 1 tool available
    ('Product_B', 'OP10'): 2,
    ('Product_A', 'OP20'): 2,
    ('Product_B', 'OP20'): 2,
}
```

---

## 🧪 Testing & Verification

### Ground Truth Test Case
The system includes a built-in verification case in `data_config.py`:
- **Demand**: 100 units each for Product A and B
- **Cycle Time**: 100 seconds (1.67 minutes) per unit
- **Available Time**: 1440 minutes (24 hours)
- **Expected Result**: Exact demand fulfillment with minimal changeovers

Run verification:
```bash
python main.py
```

Expected output:
```
Success! Bottleneck Workload: 300.0 min
24h Utilization: 20.8%

--- Detailed Allocation Result ---
     Unit    Product Operation  Quantity        Type
0  Unit_1  Product_A      OP10     100.0  Production
1  Unit_2  Product_B      OP10     100.0  Production
2  Unit_3  Product_B      OP20     100.0  Production
3  Unit_4  Product_A      OP20     100.0  Production
```

---

## 🔧 Troubleshooting

### Issue: Infeasible Solution
**Symptoms**: `Optimization Failed` or high unmet demand

**Possible Causes:**
1. Insufficient WIP for first operation
2. Tool capacity too restrictive
3. Equipment time capacity exceeded
4. Flow constraints blocking production

**Solutions:**
- Increase WIP quantities
- Add more tools or equipment
- Verify cycle times are realistic
- Check operation sequence logic

### Issue: Excessive Changeovers
**Symptoms**: Many CHANGEOVER entries in Gantt chart

**Solutions:**
- Increase `p_assign` penalty weight
- Reduce `p_continuation` if too restrictive
- Review equipment model assignments

### Issue: Unbalanced Equipment Load
**Symptoms**: Some equipment idle while others overloaded

**Solutions:**
- Implement load balancing constraints
- Adjust tool distribution
- Review WIP allocation strategy

---

## 📈 Performance Optimization

For large-scale problems (100+ products, 50+ equipment):

1. **Use Commercial Solvers**: Replace CBC with Gurobi or CPLEX
2. **Decomposition**: Split by product family or time window
3. **Heuristics**: Add initial solution hints
4. **Parallel Processing**: Enable multi-threading in solver

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📞 Support

For questions or issues:
- GitHub Issues: [https://github.com/relighting123/pjt_optimizer/issues](https://github.com/relighting123/pjt_optimizer/issues)
- Email: support@example.com

---

## 🙏 Acknowledgments

Built with:
- [PuLP](https://coin-or.github.io/pulp/) - Linear Programming
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API Framework
- [Streamlit](https://streamlit.io/) - Data Apps
- [Plotly](https://plotly.com/) - Interactive Visualizations
