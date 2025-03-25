import argparse
import os
from market_simulation import MarketSimulation

def main():
    """
    Main function to run the market simulation with configurable parameters.
    """
    parser = argparse.ArgumentParser(description='Run a Stackelberg market simulation with LLM agents')
    
    # Market parameters
    parser.add_argument('--beta', type=float, default=0.10, help='Price sensitivity coefficient (higher = more price-sensitive consumers)')
    parser.add_argument('--demand', type=float, default=1000, help='Total market demand (units)')
    parser.add_argument('--company-a-cost', type=float, default=50, help="Company A's production cost per unit (initial leader)")
    parser.add_argument('--company-b-cost', type=float, default=50, help="Company B's production cost per unit (initial follower)")
    
    # Initial market share and persistence
    parser.add_argument('--initial-market-share-a', type=float, default=0.5, 
                      help='Initial market share for Company A (0-1)')
    parser.add_argument('--persistence-factor', type=float, default=0.5,
                      help='Market share persistence factor (0-1): higher values mean more persistent market shares')
    
    # Personality setting
    parser.add_argument('--personality', type=str, choices=['neutral', 'aggressive', 'active'], default='neutral',
                      help='Agent personality: neutral (balanced), aggressive (market dominance), or active (experimental)')
    
    # Simulation parameters
    parser.add_argument('--periods', type=int, default=10, help='Number of simulation periods')
    parser.add_argument('--model', type=str, default='llama3.1:8b', 
                        help='LLM model to use (e.g., llama3.1:8b, deepseek-r1:7b)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to store simulation results')
    
    args = parser.parse_args()
    
    # Display model information
    print(f"Using LLM model: {args.model}")
    if ":" in args.model:
        provider, model_name = args.model.split(":", 1)
        print(f"Provider: {provider}, Model: {model_name}")
    else:
        print(f"Using default Ollama API with model {args.model}")
    
    # Display key simulation parameters
    print(f"Using {args.personality.upper()} personality for agents")
    print(f"Market share persistence factor: {args.persistence_factor}")
    print(f"Initial market share: Company A: {args.initial_market_share_a:.1%}, Company B: {(1-args.initial_market_share_a):.1%}")
    print(f"Price sensitivity (beta): {args.beta}")
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run the simulation
    simulation = MarketSimulation(
        beta=args.beta,
        total_demand=args.demand,
        company_a_cost=args.company_a_cost,
        company_b_cost=args.company_b_cost,
        num_periods=args.periods,
        llm_model=args.model,
        log_dir=args.output_dir,
        initial_market_share_a=args.initial_market_share_a,
        persistence_factor=args.persistence_factor,
        personality=args.personality
    )
    
    simulation.run_simulation()
    simulation.visualize_results()
    
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()
