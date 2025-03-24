# EV-Scheduler: Multi-Objective Optimization for Microgrid Sizing with Electric Vehicle Integration

This project implements a multi-objective optimization model based on MILP (Mixed-Integer Linear Programming) for the sizing of microgrids integrating renewable and non-renewable Distributed Energy Resources (DERs), Battery Energy Storage Systems (BESS), Thermal Generators (TG), Electric Vehicle Charging Stations (EVCS), and Vehicle-to-Grid (V2G) strategies.

## Overview

The proposed model focuses on three main objectives:
- **Economic:** Minimize the overall system costs by reducing both CAPEX (capital expenditure) and OPEX (operational expenditure), including the cost of the V2G service.
- **Operational:** Reduce the idle time of electric vehicles (EVs) by optimizing charging schedules and efficient use of charging stations.
- **Environmental:** Lower greenhouse gas (GHG) emissions by enhancing renewable energy penetration and strategically utilizing energy storage and V2G.

The model incorporates linearization techniques to reduce computational complexity—such as avoiding excessive use of binary variables in representing charging and discharging operations of BESS and EVs.

## Methodology

The optimization framework consists of:
- **MILP Formulation:** The problem is defined as minimizing the sum of CAPEX and discounted OPEX over the operational horizon. It is subject to energy balance constraints, component capacity limits (PV, BESS, TG, EVCS), and EV availability.
- **Multi-Objective Approach:** Trade-offs between cost, GHG emissions, and EV idle time are analyzed using a Pareto frontier. The optimal trade-off is achieved through a Nash Bargaining Solution (NBS) method.
- **Contingency Integration:** The model includes off-grid scenarios to simulate grid outages and other contingency events, ensuring system resilience through coordinated operation of BESS and V2G.

## Test Case

The test case considers:
- A 20-year operational horizon with an annual cost growth rate.
- Detailed operational parameters (maximum EDS capacity, export limits, peak load demand, curtailment penalties, etc.).
- Technical and economic data for the components (PV, BESS, TG, EVCS) based on market studies.
- Multiple scenarios for PV generation and load demand variations, including off-grid contingency events.

For detailed parameter definitions and EV scheduling, please refer to the tables provided in the original documentation.

## Results and Conclusions

Key findings from the simulation include:
- A V2G service price of 0.06 USD/kWh effectively incentivizes EV owner participation while balancing the operator's costs.
- Increased EV idle time can reduce the need for additional charging infrastructure without compromising system reliability.
- Lowering GHG emissions requires higher investments in renewable energy and storage, impacting overall system costs.
- The multi-objective equilibrium (Nash Equilibrium) reached a configuration with a total system cost of 1.06 MUSD, annual GHG emissions of 9.91 tons CO₂, and an average EV idle time of 33.75 minutes.

These results demonstrate the model’s viability for supporting strategic decisions in planning smart, sustainable microgrids.

## Repository and Reproducibility

The code, data, and models used in this study are available in this repository to facilitate reproducibility and further research. Simulations were executed using the Gurobi 11.0.1 solver with a 10% MIP gap on a machine equipped with an Intel® Core™ i7-13700 processor and 16 GB RAM.

## Installation and Usage

### Requirements
- Python 3.7+
- Libraries: NumPy, Pandas, Matplotlib, Gurobi (or any compatible MILP solver)
- Additional packages as listed in the `requirements.txt` file

### Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/TeradaZenichi/ev-scheduler.git
   cd ev-scheduler

2. Install the dependencies:
  ```bash
  pip install -r requirements.txt
```
3. Configure the MILP solver (e.g., Gurobi) according to its official documentation.
  
4. To run the optimization and generate the result graphs, simply execute:
```bash
python linearmodel.py
```

## Contributing

Contributions, suggestions, and bug reports are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).



