# Causal Impact of Weather on Bike Demand

## Summary
Bad weather reduces demand by ~39% (vs ~60% naive estimate).
Effect varies by user type and time of day.

## Key Insights
- Casual users: −51% vs Registered: −36%
- Early morning: −29% vs Off-peak: −48%

## Approach
- Fixed effects model controlling for hour of day, month, year
- Comparison with naive estimates
- Segmented analysis

## Output
See PDF presentation [here](https://github.com/glangetasq/causal_weather_bike_demand/blob/ee27bbafa067ea236214743010e223cc3fc80c32/causal_weather_bike_demand.pdf)

## Data
* [Fanaee-T, H. (2013). UCI Bike Sharing Dataset](https://doi.org/10.24432/C5W894)
