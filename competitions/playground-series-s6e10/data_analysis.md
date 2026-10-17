# Data Analysis — playground-series-s6e10

Profile sample cap: **50,000 rows per CSV**

- Train sample shape: `(50000, 23)`
- Test sample shape: `(50000, 22)`
- Target candidates: `['satisfaction']`
- Selected target: `satisfaction`
- Submission/ID candidates: `['id']`

## Missing values (train sample)

| Column | Missing |
|---|---:|
| Arrival Delay in Minutes | 17 |
| Age | 0 |
| id | 0 |
| Inflight wifi service | 0 |
| Departure/Arrival time convenient | 0 |
| Ease of Online booking | 0 |
| Flight Distance | 0 |
| Gate location | 0 |
| Food and drink | 0 |
| Seat comfort | 0 |
| Online boarding | 0 |
| On-board service | 0 |
| Leg room service | 0 |
| Baggage handling | 0 |
| Inflight entertainment | 0 |
| Checkin service | 0 |
| Cleanliness | 0 |
| Departure Delay in Minutes | 0 |
| Gender | 0 |
| Customer Type | 0 |

## Dtypes

| Column | Dtype |
|---|---|
| id | int64 |
| Age | int64 |
| Flight Distance | int64 |
| Inflight wifi service | int64 |
| Departure/Arrival time convenient | int64 |
| Ease of Online booking | int64 |
| Gate location | int64 |
| Food and drink | int64 |
| Online boarding | int64 |
| Seat comfort | int64 |
| Inflight entertainment | int64 |
| On-board service | int64 |
| Leg room service | int64 |
| Baggage handling | int64 |
| Checkin service | int64 |
| Cleanliness | int64 |
| Departure Delay in Minutes | int64 |
| Arrival Delay in Minutes | float64 |
| Gender | str |
| Customer Type | str |
| Type of Travel | str |
| Class | str |
| satisfaction | bool |
