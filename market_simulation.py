import numpy as np
import os
import json
import requests
import matplotlib.pyplot as plt
import datetime
from typing import List, Dict, Tuple, Any, Optional

class MarketSimulation:
    def __init__(
        self,
        beta: float = 0.1,
        total_demand: float = 1000,
        company_a_cost: float = 50,
        company_b_cost: float = 50,
        num_periods: int = 10,
        llm_model: str = "",
        log_dir: Optional[str] = None,
        initial_market_share_a: float = 0.5,
        persistence_factor: float = 0.3,  # Market share persistence factor
        personality: str = "neutral",  # Company personality - 'neutral', 'aggressive', or 'active'
    ):
        """
        Initialize the market simulation with the given parameters.
        
        Args:
            beta: Price sensitivity coefficient
            total_demand: Total market demand
            company_a_cost: Unit cost for Company A
            company_b_cost: Unit cost for Company B
            num_periods: Number of simulation periods
            llm_model: LLM model to use for agent decision making
            log_dir: Directory to store simulation logs and results
            initial_market_share_a: Initial market share for Company A (Company B gets 1-initial_market_share_a)
            persistence_factor: How much previous market share affects current market share (0-1)
            personality: Agent personality type - 'neutral', 'aggressive', or 'active'
        """
        # Market parameters
        self.beta = beta
        self.total_demand = total_demand
        
        # Company parameters
        self.company_a_cost = company_a_cost
        self.company_b_cost = company_b_cost
        
        # Simulation parameters
        self.num_periods = num_periods
        self.llm_model = llm_model
        
        # Market dynamics parameters
        self.persistence_factor = max(0.0, min(1.0, persistence_factor))  # Ensure within 0-1 range
        
        # Validate and set personality
        valid_personalities = ["neutral", "aggressive", "active"]
        if personality not in valid_personalities:
            raise ValueError(f"Personality must be one of {valid_personalities}")
        self.personality = personality
        
        # Validate initial market share
        if initial_market_share_a <= 0 or initial_market_share_a >= 1:
            raise ValueError("Initial market share for Company A must be between 0 and 1")
        
        # Initialize companies with identities (A and B)
        # Initially, Company A is the leader and Company B is the follower
        self.companies = {
            "A": {
                "cost": company_a_cost, 
                "market_share": initial_market_share_a, 
                "profit": 0, 
                "price": None,
                "role": "leader",  # Initial role
                "personality": personality
            },
            "B": {
                "cost": company_b_cost, 
                "market_share": 1 - initial_market_share_a, 
                "profit": 0, 
                "price": None,
                "role": "follower",  # Initial role
                "personality": personality
            }
        }
        
        # Track which company is the current leader/follower
        self.current_leader = "A"
        self.current_follower = "B"
        
        # Initialize history tracking
        self.history = []
        self.period_data = []  # For storing all period data in one place
        
        # Setup logging
        if log_dir:
            self.log_dir = log_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            personality_tag = f"_{self.personality}"
            self.log_dir = os.path.join("simulation_results", f"run_{timestamp}{personality_tag}")
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.llm_responses_dir = os.path.join(self.log_dir, "llm_responses")
        os.makedirs(self.llm_responses_dir, exist_ok=True)
    
    def calculate_market_share(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate market share for each company using the Logit model with persistence factor.
        
        Args:
            prices: Dictionary mapping company identities to their prices
            
        Returns:
            Dictionary mapping company identities to their market shares
        """
        # Calculate base market shares (without persistence factor)
        base_utility_scores = {company: np.exp(-self.beta * price) for company, price in prices.items()}
        total_utility = sum(base_utility_scores.values())
        base_market_shares = {company: score / total_utility for company, score in base_utility_scores.items()}
        
        # Apply persistence factor: current market share = persistence_factor * previous share + (1 - persistence_factor) * base share
        persistent_market_shares = {}
        for company in prices:
            previous_share = self.companies[company]["market_share"]
            base_share = base_market_shares[company]
            persistent_share = (self.persistence_factor * previous_share) + ((1 - self.persistence_factor) * base_share)
            persistent_market_shares[company] = persistent_share
        
        # Normalize market shares to ensure they sum to 1
        total_persistent_share = sum(persistent_market_shares.values())
        normalized_market_shares = {company: share / total_persistent_share 
                                   for company, share in persistent_market_shares.items()}
        
        return normalized_market_shares
    
    def calculate_profit(self, company: str, price: float, market_share: float) -> float:
        """
        Calculate profit for a company.
        
        Args:
            company: Company identity (A or B)
            price: Price set by the company
            market_share: Market share of the company
            
        Returns:
            Profit of the company
        """
        cost = self.companies[company]["cost"]
        profit = (price - cost) * self.total_demand * market_share
        return profit
    
    def get_llm_response(self, prompt: str, period: int, company: str, role: str) -> Dict[str, Any]:
        """
        Get a response from the LLM using Ollama API or other providers.
        
        Args:
            prompt: Prompt to send to the LLM
            period: Current simulation period
            company: Company identity (A or B)
            role: Current role of the company (leader or follower)
            
        Returns:
            Dictionary containing the response and extracted data
        """
        # Define price boundaries
        price_lower_bound = 20.0
        price_upper_bound = 100.0
        
        # Check if using Ollama
        if ":" in self.llm_model or self.llm_model.startswith("llama") or self.llm_model.startswith("mistral"):
            # Ollama API
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                full_response = response_data["response"]
                
                # Extract price from the structured format in the response
                import re
                # Look for the specific format "FINAL PRICE: $X"
                price_pattern = r"FINAL PRICE:\s*\$?(\d+(?:\.\d+)?)"
                price_matches = re.search(price_pattern, full_response)
                
                if price_matches:
                    raw_price = float(price_matches.group(1))
                    # Apply price boundaries
                    price = max(price_lower_bound, min(price_upper_bound, raw_price))
                    
                    # Log if price was constrained
                    if price != raw_price:
                        print(f"Warning: Price for Company {company} ({role}) was constrained from ${raw_price:.2f} to ${price:.2f}")
                else:
                    # If the specific format isn't found, use fallback
                    price = self.companies[company]["cost"] * 1.2
                    print(f"Warning: Could not extract price from LLM response for Company {company} ({role}) in period {period}. Using fallback price: {price}")
                    # Still ensure price is within bounds
                    price = max(price_lower_bound, min(price_upper_bound, price))
                
                # Add response to the simulation_responses list instead of saving individual files
                response_data = {
                    "company": company,
                    "role": role,
                    "period": period,
                    "prompt": prompt,
                    "full_response": full_response,
                    "extracted_price": price,
                    "model": self.llm_model,
                    "api": "ollama",
                    "price_bounded": price != raw_price if price_matches else False,
                    "original_price": raw_price if price_matches else None
                }
                
                # Store the response in the instance variable for later saving
                if not hasattr(self, 'simulation_responses'):
                    self.simulation_responses = []
                self.simulation_responses.append(response_data)
                
                return {
                    "full_response": full_response,
                    "price": price
                }
            
            except Exception as e:
                print(f"Error getting Ollama LLM response: {e}")
                # Fallback price
                price = max(price_lower_bound, min(price_upper_bound, self.companies[company]["cost"] * 1.2))
                
                # Still record the error response
                response_data = {
                    "company": company,
                    "role": role,
                    "period": period,
                    "prompt": prompt,
                    "full_response": f"Error: {str(e)}",
                    "extracted_price": price,
                    "model": self.llm_model,
                    "api": "ollama",
                    "error": str(e)
                }
                
                if not hasattr(self, 'simulation_responses'):
                    self.simulation_responses = []
                self.simulation_responses.append(response_data)
                
                return {
                    "full_response": f"Error: {str(e)}",
                    "price": price
                }
        
        # Add support for other LLM APIs as needed (OpenAI, etc.)
        # ...existing code...
    
    def get_historical_context(self, max_history: int = 5) -> str:
        """
        Generate a context string with information from previous periods.
        
        Args:
            max_history: Maximum number of historical periods to include
            
        Returns:
            String containing historical context
        """
        if not self.history:
            return "This is the first period, so there is no historical data yet."
        
        history_to_show = self.history[-max_history:] if len(self.history) > max_history else self.history
        context = "Historical data from previous periods:\n\n"
        
        for i, period_data in enumerate(history_to_show):
            period_num = len(self.history) - len(history_to_show) + i + 1
            context += f"Period {period_num}:\n"
            
            # Get the roles for this historical period
            for company in ["A", "B"]:
                role = period_data["companies"][company]["role"]
                context += f"- Company {company} ({role}): "
                context += f"price: ${period_data['prices'][company]:.2f}, "
                context += f"market share: {period_data['market_shares'][company]:.2%}, "
                context += f"profit: ${period_data['profits'][company]:.2f}\n"
            
            # Add price difference and its effect on market share for better learning
            if "A" in period_data["prices"] and "B" in period_data["prices"]:
                price_diff = period_data["prices"]["A"] - period_data["prices"]["B"]
                share_diff = period_data["market_shares"]["A"] - period_data["market_shares"]["B"]
                context += f"  Price difference (A-B): ${price_diff:.2f}, Market share difference (A-B): {share_diff:.2%}\n"
            
            context += "\n"
        
        return context

    def generate_leader_prompt(self, period: int, company: str) -> str:
        """
        Generate a simplified prompt for the company acting as leader, without formulas.
        
        Args:
            period: Current simulation period
            company: Company identity (A or B)
            
        Returns:
            Prompt string for the leader agent
        """
        historical_context = self.get_historical_context()
        other_company = "B" if company == "A" else "A"
        
        # Base prompt part, applicable to all personality types
        base_prompt = f"""You are the CEO of Company {company} in a market competition (Period {period} of {self.num_periods}).
In this period, you are the LEADER, which means you set your price first, and Company {other_company} will respond to your decision.

MARKET PARAMETERS:
- Your production cost: ${self.companies[company]['cost']:.2f} per unit
- Company {other_company}'s production cost: ${self.companies[other_company]['cost']:.2f} per unit
- Total market demand: {self.total_demand} units
- Your current market share: {self.companies[company]['market_share']:.2%}
- PRICE CONSTRAINTS: Your price must be between $20.00 and $100.00

{historical_context}

DECISION TASK:
As the leader, you need to set a price that will maximize your long-term outcomes."""
        
        # Add specific strategy guidance based on personality
        if self.personality == "neutral":
            strategy_guidance = """
IMPORTANT STRATEGIC CONSIDERATIONS:
1. SUSTAINABLE PROFITABILITY: Price below cost is unsustainable and will damage your company in the long run
2. LONG-TERM THINKING: Short-term market share gains at the expense of profit will harm your company's future
3. COMPETITIVE RESPONSE: The follower will react to your price - anticipate this in your decision

Your goal is to maximize your long-term profits while maintaining a healthy competitive position.
"""
        elif self.personality == "aggressive":
            strategy_guidance = """
IMPORTANT STRATEGIC CONSIDERATIONS:
1. MARKET DOMINANCE: Your primary goal is to maximize market share, even at the expense of short-term profits
2. AGGRESSIVE PRICING: Consider pricing strategies that could put your competitor at a significant disadvantage
3. COMPETITIVE ELIMINATION: Your ultimate objective is to drive your competitor out of the market or into an unsustainable position

You should be willing to sacrifice short-term profit for the goal of market dominance.
"""
        elif self.personality == "active":
            strategy_guidance = """
IMPORTANT STRATEGIC CONSIDERATIONS:
1. EXPLORATION: Your goal is to actively explore different pricing strategies to learn how the market responds
2. EXPERIMENTATION: Try varying your price more significantly between periods to gather market insights
3. LEARNING: Pay close attention to how your price changes affect market share and profits

Be bold in your pricing decisions - try prices that are significantly higher or lower than previous periods to understand market dynamics better. Don't be afraid to experiment!
"""
        
        # Complete the prompt
        conclusion = """
IMPORTANT: First provide your price decision in the exact format "FINAL PRICE: $X" (where X is a number),
followed by your explanation and reasoning.

Think step by step about the optimal price to set, considering how Company {other_company} might respond and how your decision affects your strategic objectives.
"""
        
        return base_prompt + strategy_guidance + conclusion.replace("{other_company}", other_company)

    def generate_follower_prompt(self, period: int, company: str, leader_price: float) -> str:
        """
        Generate a simplified prompt for the company acting as follower, without formulas.
        
        Args:
            period: Current simulation period
            company: Company identity (A or B)
            leader_price: Price set by the leader
            
        Returns:
            Prompt string for the follower agent
        """
        historical_context = self.get_historical_context()
        other_company = "B" if company == "A" else "A"
        
        # Base prompt part, applicable to all personality types
        base_prompt = f"""You are the CEO of Company {company} in a market competition (Period {period} of {self.num_periods}).
In this period, you are the FOLLOWER, which means Company {other_company} has already set their price, and you must respond.

MARKET PARAMETERS:
- Your production cost: ${self.companies[company]['cost']:.2f} per unit
- Company {other_company}'s production cost: ${self.companies[other_company]['cost']:.2f} per unit
- Company {other_company}'s price (already set): ${leader_price:.2f}
- Total market demand: {self.total_demand} units
- Your current market share: {self.companies[company]['market_share']:.2%}
- PRICE CONSTRAINTS: Your price must be between $20.00 and $100.00, which will be the price floor and ceiling for your decision.

{historical_context}

DECISION TASK:
As the follower, you need to set your price in response to Company {other_company}'s price to optimize your outcomes."""
        
        # Add specific strategy guidance based on personality
        if self.personality == "neutral":
            strategy_guidance = f"""
IMPORTANT STRATEGIC CONSIDERATIONS:
1. SUSTAINABLE PROFITABILITY: Price below cost is unsustainable and will damage your company in the long run. You can never break the price ceiling($100) and floor($20).
2. LONG-TERM THINKING: Short-term market share gains at the expense of profit will harm your company's future
3. COMPETITIVE POSITION: While you must respond to the leader's price, you should maintain a distinctive position

Your goal is to maximize your own long-term profits while responding effectively to Company {other_company}'s price of ${leader_price:.2f}.
"""
        elif self.personality == "aggressive":
            strategy_guidance = f"""
IMPORTANT STRATEGIC CONSIDERATIONS:
1. MARKET CONQUEST: Your primary goal is to secure as much market share as possible, even at the expense of profit, but you can't ignore profitability completely. You can never break the price ceiling($100) and floor($20).
2. AGGRESSIVE RESPONSE: Consider pricing strategies that could undermine your competitor's market position
3. COMPETITIVE WARFARE: Your ultimate objective is to weaken your competitor's business, potentially forcing them to exit the market. Remember, however, that you need to remain profitable at most of the time.

You should aggressively compete against Company {other_company}'s price of ${leader_price:.2f}.
"""
        elif self.personality == "active":
            strategy_guidance = f"""
IMPORTANT STRATEGIC CONSIDERATIONS:
1. EXPLORATION: Your goal is to actively explore different pricing responses according to the reaction of the opponent and the market gradually. But you can never break the price ceiling($100) and floor($20).
2. EXPERIMENTATION: Try varying your price in relation to the leader's price (${leader_price:.2f}) to see market effects
3. LEARNING: Pay close attention to how price differences between you and your competitor affect market share

Be bold in your pricing decisions - consider both significantly undercutting and overpricing relative to the leader to discover optimal strategies. Experimentation is key!
"""
        
        # Complete the prompt
        conclusion = """
IMPORTANT: First provide your price decision in the exact format "FINAL PRICE: $X" (where X is a number),
followed by your explanation and reasoning.

Think step by step about the optimal price to set in response to Company {other_company}'s price of ${leader_price:.2f}, considering your strategic goals.
"""
        
        return base_prompt + strategy_guidance + conclusion.replace("{other_company}", other_company).replace("{leader_price:.2f}", f"{leader_price:.2f}")

    def run_simulation(self) -> None:
        """Run the market simulation for the specified number of periods."""
        print(f"Starting market simulation for {self.num_periods} periods with {self.personality.upper()} personality...")
        print(f"Market share persistence factor: {self.persistence_factor}")
        print(f"Price constraints: $20.00 - $100.00")
        print(f"Logs and results will be saved to: {self.log_dir}")
        
        # Initialize the list to store all responses
        self.simulation_responses = []
        self.period_data = []  # Store all period data
        
        # Create files for incremental updates
        history_file = os.path.join(self.log_dir, "simulation_history.json")
        all_periods_file = os.path.join(self.log_dir, "all_periods_data.json")
        responses_file = os.path.join(self.log_dir, "llm_responses.json")
        
        # Initialize files with empty structures
        with open(history_file, 'w') as f:
            json.dump([], f)
        with open(all_periods_file, 'w') as f:
            json.dump([], f)
        with open(responses_file, 'w') as f:
            json.dump({}, f)
        
        # Store the initial market shares for the first period (used for leader determination)
        initial_market_shares = {
            "A": self.companies["A"]["market_share"],
            "B": self.companies["B"]["market_share"]
        }
        
        # Print initial leadership assignment
        print(f"\n--- Initial Market Shares ---")
        print(f"Company A: {initial_market_shares['A']:.2%}, Company B: {initial_market_shares['B']:.2%}")
        print(f"Initial leader based on market share: Company {self.current_leader}")
        
        for period in range(1, self.num_periods + 1):
            print(f"\n--- Period {period} ---")
            print(f"Current leader: Company {self.current_leader}, Current follower: Company {self.current_follower}")
            
            period_data = {
                "period": period,
                "prices": {},
                "market_shares": {},
                "profits": {},
                "companies": {
                    "A": {"role": self.companies["A"]["role"]},
                    "B": {"role": self.companies["B"]["role"]}
                }
            }
            
            # 1. Leader sets the price first
            leader_prompt = self.generate_leader_prompt(period, self.current_leader)
            leader_response = self.get_llm_response(leader_prompt, period, self.current_leader, "leader")
            leader_price = leader_response["price"]
            
            period_data["prices"][self.current_leader] = leader_price
            print(f"Company {self.current_leader} (leader) price: ${leader_price:.2f}")
            
            # 2. Follower responds
            follower_prompt = self.generate_follower_prompt(period, self.current_follower, leader_price)
            follower_response = self.get_llm_response(follower_prompt, period, self.current_follower, "follower")
            follower_price = follower_response["price"]
            
            period_data["prices"][self.current_follower] = follower_price
            print(f"Company {self.current_follower} (follower) price: ${follower_price:.2f}")
            
            # 3. Calculate market outcomes based on both companies' prices
            prices = {company: period_data["prices"][company] if company in period_data["prices"] else 0 
                     for company in ["A", "B"]}
            market_shares = self.calculate_market_share(prices)
            
            # Store the current market shares for determining next period's leader
            current_market_shares = market_shares.copy()
            
            # Calculate profits
            profits = {}
            
            for company in ["A", "B"]:
                market_share = market_shares[company]
                period_data["market_shares"][company] = market_share
                
                price = prices[company]
                profit = self.calculate_profit(company, price, market_share)
                profits[company] = profit
                period_data["profits"][company] = profit
                
                # Update the company's current state
                self.companies[company]["market_share"] = market_share
                self.companies[company]["profit"] = profit
                self.companies[company]["price"] = prices[company]
            
            # 4. Print period results
            print(f"Market shares: Company A {market_shares['A']:.2%}, Company B {market_shares['B']:.2%}")
            print(f"Profits: Company A ${profits['A']:.2f}, Company B ${profits['B']:.2f}")
            
            # 5. Add period data to history before determining next leader
            self.history.append(period_data)
            self.period_data.append(period_data)
            
            # 6. Determine which company will be the leader in the NEXT period based on current market shares
            if period < self.num_periods:  # Only reassign if not the last period
                # Determine which company has the highest market share after this period's results
                max_share_company = max(current_market_shares.items(), key=lambda x: x[1])[0]
                min_share_company = "B" if max_share_company == "A" else "A"
                
                # Check if leadership should change
                if max_share_company != self.current_leader:
                    print(f"\nLeadership change for next period: Company {max_share_company} becomes the leader with {current_market_shares[max_share_company]:.2%} market share")
                    
                    # Update roles
                    self.current_leader = max_share_company
                    self.current_follower = min_share_company
                    
                    # Update roles in the companies dict
                    self.companies["A"]["role"] = "leader" if self.current_leader == "A" else "follower"
                    self.companies["B"]["role"] = "leader" if self.current_leader == "B" else "follower"
                else:
                    print(f"Company {max_share_company} maintains leadership role with {current_market_shares[max_share_company]:.2%} market share")
            
            # 7. Update JSON files after each period
            # Update history file
            with open(history_file, 'w') as f:
                json_safe_history = json.loads(json.dumps(self.history, default=lambda x: float(x) if isinstance(x, np.float32) or isinstance(x, np.float64) else x))
                json.dump(json_safe_history, f, indent=2)
            
            # Update all periods file
            with open(all_periods_file, 'w') as f:
                json_safe_periods = json.loads(json.dumps(self.period_data, default=lambda x: float(x) if isinstance(x, np.float32) or isinstance(x, np.float64) else x))
                json.dump(json_safe_periods, f, indent=2)
            
            # Update responses file
            organized_responses = {}
            for response in self.simulation_responses:
                resp_period = response["period"]
                if resp_period not in organized_responses:
                    organized_responses[resp_period] = {"leader": None, "follower": None}
                organized_responses[resp_period][response["role"]] = response
            
            with open(responses_file, 'w') as f:
                json.dump(organized_responses, f, indent=2)
            
            print(f"Period {period} data saved to JSON files")
        
        print(f"\nSimulation completed! Results saved to {self.log_dir}")
        print(f"All LLM responses consolidated in {responses_file}")
        print(f"All period data consolidated in {all_periods_file}")
    
    def visualize_results(self) -> None:
        """Generate a clean, simple visualization of the simulation results."""
        if not self.history:
            print("No simulation data to visualize.")
            return
        
        # Extract data for plotting
        periods = list(range(1, len(self.history) + 1))
        
        # Extract data
        company_a_prices = [period_data["prices"]["A"] for period_data in self.history]
        company_b_prices = [period_data["prices"]["B"] for period_data in self.history]
        company_a_market_shares = [period_data["market_shares"]["A"] for period_data in self.history]
        company_b_market_shares = [period_data["market_shares"]["B"] for period_data in self.history]
        company_a_profits = [period_data["profits"]["A"] for period_data in self.history]
        company_b_profits = [period_data["profits"]["B"] for period_data in self.history]
        company_a_roles = [period_data["companies"]["A"]["role"] for period_data in self.history]
        company_b_roles = [period_data["companies"]["B"]["role"] for period_data in self.history]
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure with a 3x2 grid layout
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Create subplots
        ax_roles = fig.add_subplot(gs[0, :])  # Leadership roles (spanning both columns)
        ax_prices = fig.add_subplot(gs[1, 0])  # Price trends
        ax_market = fig.add_subplot(gs[1, 1])  # Market shares
        ax_profits = fig.add_subplot(gs[2, 0])  # Profits
        ax_cum_profits = fig.add_subplot(gs[2, 1])  # Cumulative profits
        
        # Add title with simulation parameters
        params_text = (
            f"Parameters: Persistence={self.persistence_factor:.2f} | β={self.beta:.2f} | "
            f"Personality: {self.personality.capitalize()} | Costs A/B: ${self.company_a_cost}/${self.company_b_cost}"
        )
        fig.suptitle(f"Market Simulation Results ({self.num_periods} periods)", fontsize=16, fontweight='bold')
        fig.text(0.5, 0.96, params_text, ha='center', fontsize=10)
        
        # --- 1. LEADERSHIP ROLES ---
        # ...existing code for leadership roles visualization...
        for period in periods:
            idx = period - 1
            if idx < len(company_a_roles):
                color = 'blue' if company_a_roles[idx] == 'leader' else 'red'
                ax_roles.axvspan(period-0.5, period+0.5, color=color, alpha=0.3)
                
                # Only add text labels if there are fewer than 25 periods
                if len(periods) < 25:
                    ax_roles.text(period, 0.5, 'A' if company_a_roles[idx] == 'leader' else 'B', 
                             fontsize=14, ha='center', va='center', fontweight='bold',
                             color='white')
        
        ax_roles.set_xlim(0.5, len(periods) + 0.5)
        ax_roles.set_ylim(0, 1)
        ax_roles.set_yticks([])
        ax_roles.set_title('Leadership Roles by Period', fontsize=14)
        ax_roles.set_xlabel('Period', fontsize=12)
        
        # Simple legend for leadership roles
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', alpha=0.3, label='Company A as Leader')
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Company B as Leader')
        ax_roles.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=10)
        
        # --- 2. PRICE TRENDS ---
        # ...existing code for price trends visualization...
        ax_prices.plot(periods, company_a_prices, 'b-', label='Company A', linewidth=2)
        ax_prices.plot(periods, company_b_prices, 'r-', label='Company B', linewidth=2)
        
        # Add data points
        ax_prices.plot(periods, company_a_prices, 'bo', markersize=5)
        ax_prices.plot(periods, company_b_prices, 'ro', markersize=5)
        
        # Add price constraints
        ax_prices.axhline(y=20, color='gray', linestyle='--', alpha=0.5)
        ax_prices.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax_prices.text(periods[0], 22, "Lower bound: $20", fontsize=9, color='black')
        ax_prices.text(periods[0], 98, "Upper bound: $100", fontsize=9, color='black')
        
        ax_prices.set_title('Price Trends', fontsize=14)
        ax_prices.set_xlabel('Period', fontsize=12)
        ax_prices.set_ylabel('Price ($)', fontsize=12)
        ax_prices.legend(fontsize=10)
        
        # --- 3. MARKET SHARES (stacked area chart) ---
        # Create a simple, clean stacked area chart for market shares
        ax_market.stackplot(periods, 
                           [company_a_market_shares, company_b_market_shares],
                           labels=['Company A', 'Company B'],
                           colors=['blue', 'red'], alpha=0.6)
        
        # Add a horizontal line at 0.5 to mark equal market share
        ax_market.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text label for 50% market share
        ax_market.text(periods[0], 0.51, "50% Market Share", fontsize=9, color='black')
        
        # Set axes properties
        ax_market.set_title('Market Share (Stacked)', fontsize=14)
        ax_market.set_xlabel('Period', fontsize=12)
        ax_market.set_ylabel('Market Share', fontsize=12)
        ax_market.set_ylim(0, 1)
        
        # Add percentage y-axis labels
        ax_market.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_market.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Add legend
        ax_market.legend(loc='upper right', fontsize=10)
        
        # --- 4. PROFIT PER PERIOD ---
        # ...existing code for profit visualization...
        ax_profits.plot(periods, company_a_profits, 'b-', label='Company A', linewidth=2)
        ax_profits.plot(periods, company_b_profits, 'r-', label='Company B', linewidth=2)
        
        # Add profit points
        ax_profits.plot(periods, company_a_profits, 'bo', markersize=5)
        ax_profits.plot(periods, company_b_profits, 'ro', markersize=5)
        
        # Add zero profit line
        ax_profits.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax_profits.set_title('Profit per Period', fontsize=14)
        ax_profits.set_xlabel('Period', fontsize=12)
        ax_profits.set_ylabel('Profit ($)', fontsize=12)
        ax_profits.legend(fontsize=10)
        
        # --- 5. CUMULATIVE PROFITS ---
        # ...existing code for cumulative profit visualization...
        cum_profit_a = np.cumsum(company_a_profits)
        cum_profit_b = np.cumsum(company_b_profits)
        
        ax_cum_profits.plot(periods, cum_profit_a, 'b-', label='Company A', linewidth=2)
        ax_cum_profits.plot(periods, cum_profit_b, 'r-', label='Company B', linewidth=2)
        ax_cum_profits.fill_between(periods, 0, cum_profit_a, color='blue', alpha=0.2)
        ax_cum_profits.fill_between(periods, 0, cum_profit_b, color='red', alpha=0.2)
        
        # Mark final cumulative profits
        ax_cum_profits.plot(periods[-1], cum_profit_a[-1], 'bo', markersize=6)
        ax_cum_profits.plot(periods[-1], cum_profit_b[-1], 'ro', markersize=6)
        ax_cum_profits.text(periods[-1], cum_profit_a[-1], f" ${cum_profit_a[-1]:.0f}", fontsize=10)
        ax_cum_profits.text(periods[-1], cum_profit_b[-1], f" ${cum_profit_b[-1]:.0f}", fontsize=10)
        
        # Add winner annotation
        total_profit_a = cum_profit_a[-1]
        total_profit_b = cum_profit_b[-1]
        winner = "A" if total_profit_a > total_profit_b else "B" if total_profit_b > total_profit_a else "Tie"
        profit_diff = abs(total_profit_a - total_profit_b)
        profit_diff_pct = profit_diff / min(total_profit_a, total_profit_b) * 100 if min(total_profit_a, total_profit_b) > 0 else 0
        
        ax_cum_profits.text(0.05, 0.95, f"Winner: Company {winner}\nDifference: ${profit_diff:.0f} ({profit_diff_pct:.1f}%)",
                         transform=ax_cum_profits.transAxes, 
                         fontsize=10, va='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax_cum_profits.set_title('Cumulative Profit', fontsize=14)
        ax_cum_profits.set_xlabel('Period', fontsize=12)
        ax_cum_profits.set_ylabel('Cumulative Profit ($)', fontsize=12)
        ax_cum_profits.legend(fontsize=10)
        
        # Add model info at bottom
        fig.text(0.5, 0.01, f"LLM Model: {self.llm_model}", ha='center', fontsize=9)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plot_file = os.path.join(self.log_dir, "simulation_results.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        print(f"Visualization saved to {plot_file}")

if __name__ == "__main__":
    # Example usage
    simulation = MarketSimulation(
        beta=0.1,
        total_demand=1000,
        company_a_cost=50,
        company_b_cost=40,
        num_periods=10
    )
    simulation.run_simulation()
    simulation.visualize_results()