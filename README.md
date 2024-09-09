# Portfolio Builder Simulator
This repository contains a multi-threaded Python application designed to simulate and optimize stock portfolios based on historical data. The program leverages popular portfolio-building algorithms such as the Universal Portfolio and Exponential Gradient to predict optimal investments. It is built to help investors backtest various portfolio strategies based on specific parameters, such as stock selection, time intervals, and risk tolerance.

# Overview
The simulator analyzes stock market data to create and manage a portfolio of stocks, optimizing returns while balancing risk. The simulation runs for a specified period, where stock data is fetched, portfolios are constructed, and performance is tracked based on real-time market movements.

# Key Features
* Multi-threaded Processing: The program uses multiple threads to handle the fetching, processing, and analysis of stock data concurrently, speeding up calculations.
* Universal Portfolio Algorithm: Implements the Universal Portfolio algorithm to predict optimal asset allocation based on historical data.
* Exponential Gradient Algorithm: Provides a second algorithm that emphasizes recent performance in asset allocation predictions, allowing for dynamic portfolio adjustments.
* Historical Data Fetching: Uses APIs like yfinance or pandas-datareader to retrieve historical stock prices, volume, and other relevant data for simulations.
* Risk-Adjusted Return Calculation: Simulates various portfolio strategies, adjusting for risk tolerance and volatility using standard metrics like Sharpe Ratio.
* Exception Handling: Built-in error management to handle API rate limits, missing data, or internet issues during data fetching.

# How It Works
1) Data Fetching: The program pulls historical data for selected stocks at random intervals based on user-specified tickers and periods (start/end date).
2) Portfolio Simulation: Stocks are evaluated using multiple algorithms, each trying to maximize portfolio performance while minimizing risk.
3) Multi-threading: Each stock’s data is fetched and processed on a separate thread to ensure the program runs efficiently and concurrently.
4) Algorithmic Decision Making:
* Universal Portfolio: Allocates assets in such a way that past-performing stocks have a higher weight, gradually learning the best strategy over time.
* Exponential Gradient: Uses a learning rate to shift weights more rapidly towards higher-performing stocks.
5) Performance Monitoring: The program periodically checks the portfolio’s status and prints out a performance report, including total returns, volatility, and risk metrics.

# Notes
The simulator is sensitive to internet connectivity issues, and API limitations may affect data fetching. Ensure the environment has sufficient network access and Python libraries (yfinance, numpy, etc.) installed.
