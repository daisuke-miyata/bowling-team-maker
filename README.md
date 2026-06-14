# Bowling Team Optimizer

Generate fair and balanced bowling teams using mixed-integer optimization.

This tool automatically assigns players to teams while balancing skill levels and gender distribution. It is designed for company events, club activities, and recreational tournaments where manual team assignment often leads to uneven teams.

---

## Features

* Automatic team generation
* Balanced team averages based on player scores
* Balanced gender distribution across teams
* Automatic determination of the number of teams
* Supports team sizes of 3 or 4 players
* CSV input and output
* Mixed-integer optimization using Pyomo

---

## Input Format

Prepare a CSV file containing player information:

```csv
name,gender,score
Alice,F,120
Bob,M,145
Charlie,M,132
Diana,F,118
```

### Columns

| Column | Description           |
| ------ | --------------------- |
| name   | Player name           |
| gender | M or F                |
| score  | Average bowling score |

---

## Optimization Model

The optimizer minimizes team imbalance while satisfying practical event constraints.

### Objectives

* Balance total team scores
* Balance gender distribution

### Constraints

* Each player belongs to exactly one team
* Team size must be 3 or 4 players
* No 5-player teams
* Number of teams is automatically selected
* Maximum number of teams can be specified

---

## Installation

Clone the repository:

```bash
git clone git@github.com:daisuke-miyata/bowling-team-optimizer.git
cd bowling-team-optimizer
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Solver Requirements

This project requires a Mixed-Integer Linear Programming (MILP) solver.

Recommended options:

* HiGHS
* CBC

Example installation for HiGHS:

```bash
pip install highspy
```

Then set:

```python
SOLVER = "highs"
```

in the source code.

---

## Usage

Place your player list in a CSV file and run:

```bash
python src/team_maker.py
```

Generated files:

* `assignment.csv` — player-to-team assignments
* `teams.csv` — team summaries and statistics

---

## Example Output

| Team   | Players             |
| ------ | ------------------- |
| Team 1 | Alice, Bob, Charlie |
| Team 2 | Diana, Eric, Frank  |
| Team 3 | Grace, Henry, Ian   |

The actual assignment is optimized to balance score and gender distribution.

---

## Motivation

Bowling events are common in companies, universities, and social clubs. Team assignment is often performed manually, which can unintentionally create large differences in team strength.

This project demonstrates how mathematical optimization can be used to automate a practical real-world task and generate fairer teams with minimal effort.

---

## License

MIT License
