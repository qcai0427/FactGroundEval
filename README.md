# Market Simulation Using LLM Agents in a Stackelberg Game

This project simulates market competition using a Stackelberg game framework with Large Language Model (LLM) agents acting as company decision-makers. It models the sequential decision-making process where a leader company sets its price first, followed by the follower company, with dynamic role transitions based on market performance.

## Overview

The simulation demonstrates how LLM agents can make strategic pricing decisions in a competitive market environment, considering factors such as:

- Profit maximization
- Market share dynamics
- Influence accumulation and decay
- Dynamic leader reassignment based on market performance

## Features

- **Mathematical Models**: Implements modified Logit model for market share calculation, profit computation, and influence dynamics.
- **LLM Agent Integration**: Uses Ollama API to leverage language models for strategic decision-making.
- **Dynamic Leadership**: Companies can switch between leader and follower roles based on market share.
- **Comprehensive Metrics**: Tracks prices, market shares, profits, and influence over time.
- **Visualization**: Generates graphs showing the evolution of key metrics with clear indications of role transitions.
- **Historical Context**: Provides agents with information about past periods (up to 3 rounds) to inform decisions.
- **Consolidated Data**: Stores all simulation data in organized JSON files for easy analysis.

## Requirements

- Python 3.7+
- Ollama (running locally on port 11434)
- Required Python packages:
  - numpy
  - matplotlib
  - requests

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llmagent2.git
   cd llmagent2
   ```

2. Install required packages:
   ```
   pip install numpy matplotlib requests
   ```

3. Make sure Ollama is running with the desired model:
   ```
   ollama run llama3.1:8b
   ```

## Usage

### Basic Usage

Run the simulation with default parameters:

```
python main.py
```

### Custom Parameters

Customize the simulation using command-line arguments:

```
python main.py --beta 0.15 --company-a-cost 60 --company-b-cost 45 --periods 20 --model llama3.1:8b
```

### Available Parameters

- `--beta`: Price sensitivity coefficient (default: 0.1)
- `--demand`: Total market demand (default: 1000)
- `--company-a-cost`: Company A's production cost (default: 50, initial leader)
- `--company-b-cost`: Company B's production cost (default: 40, initial follower)
- `--alpha`: Influence growth coefficient (default: 0.05)
- `--delta`: Influence decay rate (default: 0.1)
- `--profit-weight`: Weight for profit in leader's scoring (default: 0.8)
- `--influence-weight`: Weight for influence in leader's scoring (default: 0.2)
- `--periods`: Number of simulation periods (default: 10)
- `--model`: LLM model to use (default: llama3.1:8b)
- `--output-dir`: Custom directory for results (default: auto-generated timestamped directory)

## Output Files

Each simulation run creates a timestamped directory containing:

1. **All Periods Data**: A single JSON file (`all_periods_data.json`) containing detailed information for all periods
2. **LLM Responses**: A consolidated JSON file (`llm_responses.json`) with all LLM interactions, organized by period
3. **Simulation History**: Complete record of the entire simulation
4. **Visualizations**: 
   - `simulation_results.png`: Graphs showing prices, market shares, profits, and influence over time
   - `leadership_transitions.png`: Visual representation of when leadership roles changed

## Data Structure

### LLM Response Format

Each agent provides their decision in the format `FINAL PRICE: $X` followed by their reasoning. This structured format ensures accurate price extraction.

Example response structure:
```json
{
  "1": {
    "leader": {
      "company": "A",
      "role": "leader",
      "period": 1,
      "full_response": "FINAL PRICE: $70.00\n\nMy reasoning is...",
      "extracted_price": 70.0
    },
    "follower": {
      "company": "B",
      "role": "follower",
      "period": 1,
      "full_response": "FINAL PRICE: $65.00\n\nI chose this price because...",
      "extracted_price": 65.0
    }
  }
}
```

### Period Data Format

The `all_periods_data.json` file contains all simulation periods in a single array:

```json
[
  {
    "period": 1,
    "prices": {"A": 70.0, "B": 65.0},
    "market_shares": {"A": 0.48, "B": 0.52},
    "profits": {"A": 9600, "B": 13000},
    "influences": {"A": 0.0, "B": 0.026},
    "companies": {
      "A": {"role": "leader"},
      "B": {"role": "follower"}
    }
  },
  {
    "period": 2,
    "prices": {"A": 68.0, "B": 63.0},
    "market_shares": {"A": 0.45, "B": 0.55},
    "profits": {"A": 8100, "B": 12650},
    "influences": {"A": 0.0, "B": 0.054},
    "companies": {
      "A": {"role": "follower"},
      "B": {"role": "leader"}
    }
  }
]
```

## Mathematical Model

### Market Share
Uses a modified Logit model:
```
s_i = e^(-β*p_i) / Σ(e^(-β*p_j))
```

### Profit
Calculated as:
```
π_i = (p_i - c_i) * D * s_i
```

### Influence
Dynamic mechanism with growth and decay:
- Growth (if highest market share): `I_{i,t} = I_{i,t-1} + α * s_i`
- Decay (otherwise): `I_{i,t} = I_{i,t-1} * (1 - δ)`

## Future Extensions

- Multi-follower Nash equilibrium
- Additional market parameters (customer loyalty, product differentiation)
- Advanced LLM reasoning capabilities
- Custom agent personalities and strategies
- Support for additional LLM providers and APIs

## License

[MIT License]

## Acknowledgments

- This project implements concepts from game theory and market competition models
- Utilizes open-source LLMs via Ollama for agent decision-making
